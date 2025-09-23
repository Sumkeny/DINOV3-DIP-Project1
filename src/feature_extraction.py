# feature_extraction.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import argparse

# Import the model from our model.py file
from model import DINOv3Backbone

class ImageDataset(Dataset):
    """A simple dataset to load images from a directory."""
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

    # 1. Model and Transformation
    model = DINOv3Backbone().to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)), # Standard Re-ID size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Dataset and DataLoader
    dataset = ImageDataset(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Feature Extraction Loop
    all_features = []
    all_paths = []
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader, desc=f"Extracting features from {os.path.basename(args.image_dir)}"):
            imgs = imgs.to(device)
            # Use the extract_global method which includes L2 normalization
            features = model.extract_global(imgs)
            
            all_features.append(features.cpu().numpy())
            all_paths.extend(paths)

    all_features = np.vstack(all_features)

    # 4. Save features
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # Use np.savez to save both features and paths for easy loading later
    np.savez(args.output_path, features=all_features, paths=np.array(all_paths))
    print(f"Extracted {len(all_features)} features. Saved to {args.output_path}")
    print(f"Feature matrix shape: {all_features.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features from images for Re-ID.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the directory containing images (e.g., 'data/market1501/query/')")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the extracted features (e.g., 'features/query_feats.npz')")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for feature extraction.")
    
    args = parser.parse_args()
    main(args)

# Example command line usage:
# python feature_extraction.py --image_dir path/to/market1501/query --output_path features/market1501/query_feats.npz
# python feature_extraction.py --image_dir path/to/market1501/bounding_box_test --output_path features/market1501/gallery_feats.npz