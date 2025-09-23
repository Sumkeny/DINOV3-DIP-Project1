#######辅助函数#######
import os
import glob
import pickle
import re
from PIL import Image
from torch.utils.data import Dataset

def load_image_paths(data_dir):
    """加载一个目录下所有 jpg 图像的路径"""
    image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
    return sorted(image_paths)

def save_features(path, features, image_paths):
    """保存特征和对应的图像路径"""
    with open(path, 'wb') as f:
        pickle.dump({'features': features, 'paths': image_paths}, f)
    print(f"Features saved to {path}")

def load_features(path):
    """加载特征和图像路径"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Features loaded from {path}")
    return data['features'], data['paths']

def parse_market1501_filename(filename):
    """从 Market-1501 文件名中解析 PID 和 CamID"""
    # 正则表达式匹配文件名，例如 0001_c1s1_000451_01.jpg
    match = re.match(r'(\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.jpg', os.path.basename(filename))
    if match:
        pid = int(match.group(1))
        camid = int(match.group(2))
        return pid, camid
    # 处理 -1 的情况 (junk images)
    elif os.path.basename(filename).startswith('-1'):
        return -1, -1
    return None, None

class ReIDDataset(Dataset):
    """用于特征提取的 PyTorch Dataset"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

class UnlabeledTrainDataset(Dataset):
    """用于 Adapter 训练的 Dataset"""
    def __init__(self, image_paths, pseudo_labels, transform=None):
        self.image_paths = image_paths
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.pseudo_labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label