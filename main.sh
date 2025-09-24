#!/bin/bash
# 这是一个修正版的脚本，用于运行完整的 Re-ID 实验流程
# set -e 会让脚本在任何命令失败时立即退出，方便调试
set -e

# --- 配置 ---
# 数据集所在的根目录
DATA_DIR="data/market1501" 
# 保存训练好的 adapter 的目录
OUTPUT_DIR="outputs"
# 保存提取出的特征文件的目录
FEATURES_DIR="features"
# 源代码所在的目录
SRC_DIR="src"

# 创建输出目录，如果它们不存在的话
mkdir -p ${OUTPUT_DIR} ${FEATURES_DIR}

# # --- 1. 基线模型测试 (Baseline) ---
# echo "--- Running Baseline Feature Extraction ---"
# # MODIFIED: 将 --batch_save_size 替换为 --batch_size
# python ${SRC_DIR}/feature_extraction.py \
#     --data_dir ${DATA_DIR} \
#     --subdir query \
#     --output_dir ${FEATURES_DIR} \
#     --output_prefix "baseline_query" \
#     --batch_size 128

# python ${SRC_DIR}/feature_extraction.py \
#     --data_dir ${DATA_DIR} \
#     --subdir bounding_box_test \
#     --output_dir ${FEATURES_DIR} \
#     --output_prefix "baseline_gallery" \
#     --batch_size 128

# echo "--- Running Baseline Evaluation ---"
# python ${SRC_DIR}/evaluate.py \
#     --feats_dir ${FEATURES_DIR} \
#     --query_prefix "baseline_query" \
#     --gallery_prefix "baseline_gallery"

# # --- 2. Adapter 训练 (UDA) ---
# echo "--- Running Adapter Training (UDA) ---"
# # 这个命令通常不需要修改，因为它调用的是另一个脚本
# python ${SRC_DIR}/unsupervised_adapt.py \
#     --data_dir "${DATA_DIR}/bounding_box_train" \
#     --output_path "${OUTPUT_DIR}/adapter.pth"

# --- 3. 适应后模型测试 (Adapted) ---
echo "--- Running Adapted Feature Extraction ---"
# MODIFIED: 将 --batch_save_size 替换为 --batch_size，并使用 --adapter_path
python ${SRC_DIR}/feature_extraction.py \
    --data_dir ${DATA_DIR} \
    --subdir query \
    --output_dir ${FEATURES_DIR} \
    --output_prefix "adapted_query" \
    --adapter_path "${OUTPUT_DIR}/adapter.pth" \
    --batch_size 128

python ${SRC_DIR}/feature_extraction.py \
    --data_dir ${DATA_DIR} \
    --subdir bounding_box_test \
    --output_dir ${FEATURES_DIR} \
    --output_prefix "adapted_gallery" \
    --adapter_path "${OUTPUT_DIR}/adapter.pth" \
    --batch_size 128

echo "--- Running Adapted Model Evaluation ---"
python ${SRC_DIR}/evaluate.py \
    --feats_dir ${FEATURES_DIR} \
    --query_prefix "adapted_query" \
    --gallery_prefix "adapted_gallery"

echo "--- Pipeline finished successfully! ---"