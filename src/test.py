# file: test_model.py
import torch
import sys

# 将 src 目录添加到 Python 路径中，这样才能找到 model 模块
sys.path.append('src')

# 现在可以从 src/model.py 导入了
from model import DINOv3ReID

def run_test():
    """独立测试 model.py 是否能正常工作"""
    print("="*20 + " Starting Model Test " + "="*20)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print("GPU Name:", torch.cuda.get_device_name(0))

        # --- 关键测试 1: 尝试创建模型实例 ---
        print("\nStep 1: Attempting to create DINOv3ReID model instance...")
        model_instance = DINOv3ReID()
        model_instance.to(device)
        print("✅ SUCCESS: Model instance created and moved to device successfully!")

        # --- 关键测试 2: 尝试一次前向传播 ---
        print("\nStep 2: Attempting a forward pass with a dummy image tensor...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output_features = model_instance(dummy_input)
        
        print("✅ SUCCESS: Forward pass completed successfully!")
        print(f"Output feature shape: {output_features.shape}")
        print(f"Expected feature shape: (1, {model_instance.feat_dim})")
        
        assert output_features.shape == (1, model_instance.feat_dim), "Output shape is incorrect!"
        print("✅ SUCCESS: Output feature shape is correct.")

    except Exception as e:
        print(f"\n❌ FAILED: An error occurred during the test.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")

    print("\n" + "="*20 + " Model Test Finished " + "="*20)

if __name__ == "__main__":
    run_test()