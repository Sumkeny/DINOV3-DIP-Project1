# file: src/feature_extraction.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import pickle
import glob
import re

# 假设 model.py 在同级目录下
from model import DINOv3ReID, AdapterHead

# --- 为批处理定义一个 PyTorch Dataset ---
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # 过滤掉非图片文件和 Market-1501 的干扰项
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png')) and not f.startswith('-1')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

def extract_and_save_features_batched(model, adapter, data_dir, subdir, output_dir, output_prefix, device, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_dir = os.path.join(data_dir, subdir)
    dataset = ImageDataset(img_dir, transform=transform)
    # 使用 DataLoader 进行高效批处理加载
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_features = []
    all_pids = []
    all_camids = []
    all_img_paths = []

    model.eval()
    if adapter:
        adapter.eval()
    
    print(f"Extracting features from {subdir} in batches...")
    
    with torch.no_grad():
        for images, img_names in tqdm(dataloader):
            images = images.to(device)
            
            backbone_feats = model(images)
            
            if adapter:
                _, final_feats = adapter(backbone_feats)
                final_feats = torch.nn.functional.normalize(final_feats, dim=1)
            else:
                final_feats = backbone_feats

            all_features.append(final_feats.cpu().numpy())
            
            # --- 这是最关键的 Bug 修复 ---
            for img_name in img_names:
                # 解析 PID
                pid = int(img_name.split('_')[0])
                
                # 使用正则表达式可靠地解析 CamID
                match = re.search(r'c(\d+)', img_name)
                if match:
                    camid = int(match.group(1))
                else:
                    # 如果解析失败，给一个默认值并警告，而不是让程序崩溃或使用错误值
                    camid = -1 
                    print(f"Warning: Could not parse CamID from {img_name}")
                
                all_pids.append(pid)
                all_camids.append(camid)
                all_img_paths.append(os.path.join(img_dir, img_name))

    # 将所有批次的特征合并成一个大的 Numpy 数组
    all_features = np.vstack(all_features)

    # 保存到一个单一的 pickle 文件中
    output_path = os.path.join(output_dir, f"{output_prefix}.pkl")
    data_to_save = {
        'img_paths': all_img_paths,
        'pids': all_pids,
        'camids': all_camids,
        'features': all_features
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print(f"Saved all features for {subdir} to {output_path}")

# --- 由于不再分批保存，evaluate.py 也需要一个小修改 ---
# 我们可以在这里直接提供新的 load_features 函数，或者修改 evaluate.py
# 为简单起见，我将在下面提供修改后的 evaluate.py

def parse_args():
    parser = argparse.ArgumentParser(description="Batched Feature Extraction")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--subdir', type=str, required=True, choices=['query', 'bounding_box_test'], help="Subdirectory to process")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the feature file")
    parser.add_argument('--output_prefix', type=str, required=True, help="Prefix for the output pkl file")
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for feature extraction")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DINOv3ReID().to(device)
    adapter = None
    if args.adapter_path:
        adapter = AdapterHead().to(device)
        adapter.load_state_dict(torch.load(args.adapter_path))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    extract_and_save_features_batched(model, adapter, args.data_dir, args.subdir, args.output_dir, args.output_prefix, device, args.batch_size)