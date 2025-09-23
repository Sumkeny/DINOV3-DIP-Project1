import torch
from model import DINOv3ReID, FEATURE_DIM # 从修正后的 model.py 导入

print("--- Testing DINOv3 Feature Extractor ---")

try:
    # 1. 初始化模型
    print("Initializing model...")
    model = DINOv3ReID()
    model.eval()
    print("Model initialized successfully.")

    # 2. 创建一个测试输入
    # 批大小=1, 通道=3, 高=256, 宽=128
    test_input = torch.randn(1, 3, 256, 128)
    print(f"Created test input with shape: {test_input.shape}")

    # 3. 进行前向传播
    print("Performing forward pass...")
    with torch.no_grad():
        output = model(test_input)
    print("Forward pass successful.")

    # 4. 检查输出形状
    print(f"Output shape: {output.shape}")
    
    # 5. 验证形状是否正确
    expected_shape = torch.Size([1, FEATURE_DIM])
    if output.shape == expected_shape:
        print(f"\nSUCCESS! The output shape {output.shape} is correct.")
    else:
        print(f"\nFAILURE! Expected shape {expected_shape}, but got {output.shape}.")

except Exception as e:
    print(f"\nAn error occurred during the test: {e}")