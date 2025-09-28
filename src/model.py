import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
import re
from glob import glob
from PIL import Image
import numpy as np

# --- 全局设备定义 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 自定义Dataset类 (步骤1) ---
class Market1501Dataset(Dataset):
    """自定义Market-1501数据集 (修正版)"""
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        
        # 先获取所有可能的图片路径
        all_paths = glob(os.path.join(data_dir, '*.jpg'))
        
        # 创建空的列表来存储有效的数据
        self.img_paths = []
        self.pids = []
        self.camids = []
        
        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        for path in all_paths:
            match = pattern.search(path)
            # 只有当文件名匹配成功时，才处理这个文件
            if match:
                pid, camid = map(int, match.groups())
                # 过滤掉 "junk" 图像 (-1 的 pid)
                if pid == -1:
                    continue
                
                # 将所有相关信息同步添加到列表中
                self.img_paths.append(path)
                self.pids.append(pid)
                self.camids.append(camid)

        # 创建从原始pid到从0开始的标签的映射
        self.unique_pids = sorted(list(set(self.pids)))
        self.pid2label = {pid: label for label, pid in enumerate(self.unique_pids)}

    def __len__(self):
        # 长度现在是基于有效图片路径的长度
        return len(self.img_paths)

    def __getitem__(self, index):
        # 因为所有列表长度都一致，所以这里的索引是安全的
        path = self.img_paths[index]
        pid_original = self.pids[index]
        camid = self.camids[index]
        
        label = self.pid2label[pid_original]
        
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label, pid_original, camid

# --- 模型加载函数 ---
def load_model():
    model_name = 'timm/vit_small_patch16_224.dino' # 使用 timm 中确认存在的 DINO 模型
    # 注意: DINOv3 可能不在timm官方库中，这里用一个标准的DINO模型代替
    # 如果你有本地的DINOv3模型，可以使用 timm.create_model('vit_small_patch16_224', pretrained=False, checkpoint_path='path/to/your/checkpoint.pth')
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("模型装载完成")
    return model

# --- 模型定义函数 (步骤3) ---
def def_model(model, num_classes):
    feature_dim = model.num_features
    classifier = nn.Sequential(nn.Linear(feature_dim, num_classes))
    class CustomDINO(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            features = self.backbone(x)
            output = self.head(features)
            return output
    custom_model = CustomDINO(model, classifier)
    print("定义训练模型完成")
    return custom_model, feature_dim

# --- 数据集加载函数 (步骤2) ---
def load_dataset(model, dataset_path):
    data_config = timm.data.resolve_model_data_config(model)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    val_transforms = timm.data.create_transform(**data_config, is_training=False)
    train_dir = os.path.join(dataset_path, 'bounding_box_train')
    query_dir = os.path.join(dataset_path, 'query')
    gallery_dir = os.path.join(dataset_path, 'bounding_box_test')
    train_dataset = Market1501Dataset(train_dir, transform=train_transforms)
    query_dataset = Market1501Dataset(query_dir, transform=val_transforms)
    gallery_dataset = Market1501Dataset(gallery_dir, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    print("装载Market-1501数据集完成")
    return train_loader, query_loader, gallery_loader, data_config, len(train_dataset.unique_pids)

# --- 模型训练函数 ---
def train_model(custom_model, train_loader):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(custom_model.head.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        custom_model.train()
        custom_model.head.train() # 确保分类头是训练模式
        running_loss = 0.0
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    print('模型训练完成')

# --- 模型评估函数 (步骤4) ---
@torch.no_grad()
def evaluate_reid(model, query_loader, gallery_loader, device):
    model.eval()
    print("开始提取查询集和检索集特征...")
    
    # 提取查询集特征
    qf, q_pids, q_camids = [], [], []
    for images, _, pids, camids in query_loader:
        # 我们只需要骨干网络提取的特征，而不是分类头的输出
        features = model.backbone(images.to(device))
        qf.append(features.cpu())
        q_pids.extend(pids.numpy())
        q_camids.extend(camids.numpy())
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    
    # 提取检索集特征
    gf, g_pids, g_camids = [], [], []
    for images, _, pids, camids in gallery_loader:
        features = model.backbone(images.to(device))
        gf.append(features.cpu())
        g_pids.extend(pids.numpy())
        g_camids.extend(camids.numpy())
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("特征提取完成，开始计算距离矩阵...")
    # L2 归一化特征
    qf = torch.nn.functional.normalize(qf, dim=1, p=2)
    gf = torch.nn.functional.normalize(gf, dim=1, p=2)
    
    # 计算余弦距离矩阵
    dist_mat = 1 - torch.matmul(qf, gf.t())
    dist_mat = dist_mat.numpy()

    print("计算mAP和Rank-k...")
    cmc = np.zeros(len(g_pids))
    ap = 0.0
    
    for i in range(len(q_pids)):
        query_pid = q_pids[i]
        query_camid = q_camids[i]
        
        # 排序并获取索引
        order = np.argsort(dist_mat[i])
        
        # --- Market-1501 评估规则 ---
        # 1. 移除与查询图像自身完全相同的图像（同一PID，同一摄像头ID）
        # 2. 忽略 "junk" 图像（PID为-1或0，但在我们的数据集中已过滤）
        remove = (g_pids[order] == query_pid) & (g_camids[order] == query_camid)
        keep = np.invert(remove)
        
        # 获取排序后，有效的 gallery 图像中的匹配情况 (True/False 数组)
        matches = (g_pids[order][keep] == query_pid)

        if not np.any(matches):
            continue # 如果没有一个匹配项，则跳过

        # --- 计算 CMC (Rank-k) ---
        first_match_idx = np.where(matches)[0][0]
        cmc[first_match_idx:] += 1

        # --- 计算 AP (Average Precision) ---
        num_rel = matches.sum()
        positions = np.where(matches)[0]
        precisions = (np.arange(1, num_rel + 1)) / (positions + 1)
        AP = precisions.mean()
        ap += AP

    # 计算最终的 mAP 和 CMC
    mAP = ap / len(q_pids)
    cmc = cmc / len(q_pids)

    print(f'评估结果: mAP = {mAP:.2%}, Rank-1 = {cmc[0]:.2%}, Rank-5 = {cmc[4]:.2%}, Rank-10 = {cmc[9]:.2%}')
    return mAP, cmc

# --- 模型保存函数 ---
def save_model(custom_model, num_classes, feature_dim, data_config):
    torch.save({
        'classifier_state_dict': custom_model.head.state_dict(),
        'num_classes': num_classes,
        'feature_dim': feature_dim,
        'data_config': data_config
    }, 'dino_market1501_head.pth')
    print("模型保存完成")

# --- 主流程 (步骤5) ---
if __name__ == '__main__':
    model = load_model()
    dataset_path = '/home/dinox/workspace/dinov3-reid-project/data/Market-1501-v15.09.15'
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集路径 '{dataset_path}' 不存在. 请下载 Market-1501 数据集并解压到此路径.")

    train_loader, query_loader, gallery_loader, data_config, num_classes = load_dataset(model, dataset_path)
    custom_model, feature_dim = def_model(model, num_classes)
    train_model(custom_model, train_loader)
    save_model(custom_model, num_classes, feature_dim, data_config)
    evaluate_reid(custom_model, query_loader, gallery_loader, device)