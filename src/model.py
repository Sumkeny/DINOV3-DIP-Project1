# file: src/model.py
import torch
import torch.nn as nn
import timm

# 确保这是 timm 库能识别的正确 DINOv3 模型名称
MODEL_NAME = 'vit_base_patch16_dinov3'
FEATURE_DIM = 768  # ViT-Base's feature dimension

class DINOv3ReID(nn.Module):
    """
    使用 timm 加载 DINOv3 作为特征提取器。
    """
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        # 使用 num_classes=0 来移除分类头，直接获取特征
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        # 自动获取特征维度
        self.feat_dim = self.backbone.num_features
        
        # 冻结骨干网络，使其作为固定的特征提取器
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        前向传播，提取并 L2 归一化特征。
        """
        with torch.no_grad():
            global_feature = self.backbone(x)
        # L2 归一化对于度量学习和检索至关重要
        return nn.functional.normalize(global_feature, dim=1)

class AdapterHead(nn.Module):
    """
    用于 UDA 微调的轻量级 Adapter。
    """
    def __init__(self, in_dim=FEATURE_DIM, hidden_dim=512, num_classes=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        if num_classes:
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc2 = None
        self.adapted_feature_extractor = nn.Sequential(self.fc1, self.relu)
        
    def forward(self, x):
        adapted_feats = self.adapted_feature_extractor(x)
        if self.fc2:
            logits = self.fc2(adapted_feats)
            return logits, adapted_feats
        else:
            return None, adapted_feats