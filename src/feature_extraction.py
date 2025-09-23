# src/feature_extraction.py (已修正导入错误)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import argparse

# 确认 model.py 在同一个目录下或 Python 路径中
# --- 这里的导入名称已被修正 ---
from model import DINOv3Backbone

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    image_dir = os.path.join(args.data_dir, args.subdir)
    output_path = os.path.join(args.output_dir, f"{args.output_prefix}_feats.npz")
    print(f"Processing images from: {image_dir}")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    model = DINOv3Backbone().to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_features = []
    all_paths = []
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader, desc=f"Extracting from {args.subdir}"):
            imgs = imgs.to(device)
            features = model.extract_global(imgs)
            all_features.append(features.cpu().numpy())
            all_paths.extend(paths)

    all_features = np.vstack(all_features)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(output_path, features=all_features, paths=np.array(all_paths))
    print(f"Extracted {len(all_features)} features. Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features for Re-ID.")
    parser.add_argument('--data_dir', type=str, required=True, help="Base directory of the dataset (e.g., '/content/datasets/market1501/')")
    parser.add_argument('--subdir', type=str, required=True, help="Subdirectory to process (e.g., 'query' or 'bounding_box_test')")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the features.")
    parser.add_argument('--output_prefix', type=str, required=True, help="Prefix for the output .npz file (e.g., 'baseline_query')")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for feature extraction.")
    
    args = parser.parse_args()
    main(args)