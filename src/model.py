# model.py
import torch
import torch.nn as nn
import timm

class DINOv3Backbone(nn.Module):
    """
    DINOv3 (as implemented by DINOv2) Backbone Wrapper.
    Uses the timm library to load a pretrained DINOv2 model.
    """
    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m', pretrained=True):
        super().__init__()
        # Load the pretrained model from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        # The feature dimension depends on the model architecture. For ViT-L, it's 1024.
        self.feat_dim = self.backbone.head.in_features
        # Remove the original classification head
        self.backbone.head = nn.Identity()

    def forward(self, x):
        """
        Extracts features from the backbone.
        By default, timm models for ViT return the CLS token feature.
        """
        return self.backbone(x)

    def extract_global(self, x):
        """
        A helper method to extract and L2-normalize the global feature.
        This is the primary method used for Re-ID feature extraction.
        """
        with torch.no_grad():
            feat = self.forward(x)
        # L2-normalize the features
        return nn.functional.normalize(feat, p=2, dim=1)
        
    def freeze_backbone(self):
        """Freeze all parameters of the backbone."""
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters of the backbone."""
        print("Unfreezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = True

class AdapterHead(nn.Module):
    """
    A simple Adapter Head for fine-tuning.
    Consists of a small MLP to adapt features to a new domain.
    """
    def __init__(self, in_dim, out_dim=512, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.head(x)

if __name__ == '__main__':
    # Example usage:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize the model
    model = DINOv3Backbone().to(device)
    model.eval()
    print(f"Model loaded on {device}. Feature dimension: {model.feat_dim}")

    # 2. Create a dummy input tensor
    dummy_input = torch.randn(4, 3, 224, 224).to(device) # DINOv2 default size
    
    # 3. Extract features
    features = model.extract_global(dummy_input)
    
    print("Input shape:", dummy_input.shape)
    print("Output feature shape:", features.shape)
    
    # 4. Check normalization
    norms = torch.norm(features, p=2, dim=1)
    print("Feature norms (should be close to 1):", norms)