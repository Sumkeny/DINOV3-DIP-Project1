import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from config import *
from dataset import ReIDDataset, get_transform
from model import DINOv3Backbone, AdapterHead

def extract_features(model, dataloader, use_adapter=False, adapter=None):
    model.eval()
    if use_adapter and adapter:
        adapter.eval()

    all_features = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for imgs, pids, camids, _ in tqdm(dataloader, desc="Extracting Features"):
            imgs = imgs.to(DEVICE)
            
            features = model(imgs)
            if use_adapter and adapter:
                features = adapter(features)

            all_features.append(features.cpu().numpy())
            all_pids.extend(pids.numpy())
            all_camids.extend(camids.numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.array(all_pids),
        np.array(all_camids),
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='e.g., market1501 or msmt17')
    parser.add_argument('--subset', type=str, required=True, help='e.g., query, gallery, or train')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to trained adapter weights')
    args = parser.parse_args()

    # 目录设置
    data_paths = {
        'market1501': {
            'query': os.path.join(DATA_DIR, 'market1501/query'),
            'gallery': os.path.join(DATA_DIR, 'market1501/bounding_box_test')
        },
        'msmt17': {
            'train': os.path.join(DATA_DIR, 'msmt17/train'),
            'query': os.path.join(DATA_DIR, 'msmt17/test/query'),
            'gallery': os.path.join(DATA_DIR, 'msmt17/test/gallery')
        }
    }
    img_dir = data_paths[args.dataset][args.subset]

    # 加载模型
    backbone = DINOv3Backbone(MODEL_NAME).to(DEVICE)
    adapter = None
    use_adapter = False
    if args.adapter_path:
        adapter = AdapterHead(in_dim=backbone.feat_dim).to(DEVICE)
        adapter.load_state_dict(torch.load(args.adapter_path))
        use_adapter = True
        print(f"Loaded adapter from {args.adapter_path}")

    # 数据加载
    transform = get_transform(IMAGE_SIZE)
    dataset = ReIDDataset(img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 特征提取
    features, pids, camids = extract_features(backbone, dataloader, use_adapter, adapter)

    # 保存结果
    output_dir = os.path.join(OUTPUT_DIR, 'features', args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    adapter_suffix = '_adapted' if use_adapter else ''
    output_path = os.path.join(output_dir, f'{args.subset}_features{adapter_suffix}.npz')
    np.savez(output_path, features=features, pids=pids, camids=camids)
    print(f"Saved features to {output_path}")