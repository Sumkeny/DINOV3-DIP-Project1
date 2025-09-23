#!/bin/bash
set -e

# ... (Configuration remains the same) ...
DATA_DIR="data/market1501"
OUTPUT_DIR="outputs"
FEATURES_DIR="features"
SRC_DIR="src"
mkdir -p ${OUTPUT_DIR} ${FEATURES_DIR}

# --- Baseline ---
echo "--- Running Baseline Feature Extraction ---"
python ${SRC_DIR}/feature_extraction.py --data_dir ${DATA_DIR} --subdir query --output_dir ${FEATURES_DIR} --output_prefix "baseline_query" --batch_save_size 3000
python ${SRC_DIR}/feature_extraction.py --data_dir ${DATA_DIR} --subdir bounding_box_test --output_dir ${FEATURES_DIR} --output_prefix "baseline_gallery" --batch_save_size 3000

echo "--- Running Baseline Evaluation ---"
python ${SRC_DIR}/evaluate.py --feats_dir ${FEATURES_DIR} --query_prefix "baseline_query" --gallery_prefix "baseline_gallery"

# --- Adapter Training ---
echo "--- Running Adapter Training (UDA) ---"
python ${SRC_DIR}/unsupervised_adapt.py --data_dir "${DATA_DIR}/bounding_box_train" --output_path "${OUTPUT_DIR}/adapter.pth"

# --- Adapted ---
echo "--- Running Adapted Feature Extraction ---"
python ${SRC_DIR}/feature_extraction.py --data_dir ${DATA_DIR} --subdir query --output_dir ${FEATURES_DIR} --output_prefix "adapted_query" --adapter_path "${OUTPUT_DIR}/adapter.pth" --batch_save_size 4000
python ${SRC_DIR}/feature_extraction.py --data_dir ${DATA_DIR} --subdir bounding_box_test --output_dir ${FEATURES_DIR} --output_prefix "adapted_gallery" --adapter_path "${OUTPUT_DIR}/adapter.pth" --batch_save_size 4000

echo "--- Running Adapted Model Evaluation ---"
python ${SRC_DIR}/evaluate.py --feats_dir ${FEATURES_DIR} --query_prefix "adapted_query" --gallery_prefix "adapted_gallery"

echo "--- Pipeline finished successfully! ---"