# file: model.py

import torch
import torch.nn as nn
import timm

# --- 模型配置 (使用 timm 加载 DINOv3 ViT-Base) ---
# 这是正确的 DINOv3 模型名称
MODEL_NAME = 'vit_base_patch16_dinov3.lvd1689m'

FEATURE_DIM = 768  # ViT-Base's feature dimension

class DINOv3ReID(nn.Module):
 
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        
        # 使用 num_classes=0 来移除分类头，直接获取特征
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        # 自动获取特征维度
        self.feat_dim = self.backbone.num_features
        
        # 冻结骨干网络
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            global_feature = self.backbone(x)

        # L2 归一化，对于度量学习和检索至关重要
        return nn.functional.normalize(global_feature, dim=1)

class AdapterHead(nn.Module):

    def __init__(self, in_dim=FEATURE_DIM, hidden_dim=512, num_classes=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 如果需要分类（例如使用伪标签），则添加分类头
        if num_classes:
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc2 = None
            
        # 仅提取适应后的特征
        self.adapted_feature_extractor = nn.Sequential(self.fc1, self.relu)
        
    def forward(self, x):
        adapted_feats = self.adapted_feature_extractor(x)
        # 如果有分类头，同时返回 logits 和特征
        if self.fc2:
            logits = self.fc2(adapted_feats)
            return logits, adapted_feats
        else:
            # 否则只返回适应后的特征
            return None, adapted_feats