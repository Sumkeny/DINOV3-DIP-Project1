import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ReIDDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        
        # 正则表达式用于解析文件名, 例如: 0001_c1s1_001051_00.jpg
        self.pid_pattern = re.compile(r'([-\d]+)_c(\d)')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        filename = os.path.basename(img_path)
        match = self.pid_pattern.match(filename)
        if match:
            pid, camid = map(int, match.groups())
            # Market1501 的-1是干扰项
            if pid == -1: pid = 99999 
        else: # for unlabeled data
            pid, camid = -1, -1

        return img, pid, camid, img_path

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# UDA 训练时使用的数据集
class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, pseudo_labels, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.pseudo_labels = pseudo_labels # 这是一个numpy array
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.pseudo_labels[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label