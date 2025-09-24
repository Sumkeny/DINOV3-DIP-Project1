import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

from model import DINOv3ReID, AdapterHead

def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Domain Adaptation")
    parser.add_argument('--data_dir', type=str, default='data/market1501/bounding_box_train', help="Path to unlabeled target dataset")
    parser.add_argument('--num_clusters', type=int, default=751, help="Number of clusters for KMeans (num identities in Market-1501 train)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train adapter")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--output_path', type=str, default='outputs/adapter.pth', help="Path to save the trained adapter")
    return parser.parse_args()

class UnlabeledDataset(Dataset):
    def __init__(self, root, img_paths, pseudo_labels, transform):
        self.root = root
        self.img_paths = img_paths
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_paths[idx])
        label = self.pseudo_labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Extract features for the unlabeled dataset
    print("Step 1: Extracting initial features for clustering...")
    model = DINOv3ReID().to(device)
    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_paths = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.jpg')])
    all_features = []
    # Batch processing for faster feature extraction
    batch_size = 128
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Extracting"):
        batch_paths = img_paths[i:i+batch_size]
        batch_imgs = [Image.open(os.path.join(args.data_dir, p)).convert('RGB') for p in batch_paths]
        batch_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            feats = model(batch_tensors)
        all_features.append(feats.cpu().numpy())
    all_features = np.vstack(all_features)
    
    # 2. Perform clustering to get pseudo-labels
    print(f"Step 2: Clustering {len(all_features)} features into {args.num_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10, verbose=0).fit(all_features)
    pseudo_labels = kmeans.labels_

    # 3. Train the adapter
    print("Step 3: Training the adapter...")
    adapter = AdapterHead(num_classes=args.num_clusters).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = UnlabeledDataset(args.data_dir, img_paths, pseudo_labels, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0
        for imgs, p_labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, p_labels = imgs.to(device), p_labels.to(device)
            
            backbone_feats = model(imgs)
            logits, _ = adapter(backbone_feats)
            
            loss = criterion(logits, p_labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(adapter.state_dict(), args.output_path)
    print(f"Adapter saved to {args.output_path}")

if __name__ == '__main__':
    main()