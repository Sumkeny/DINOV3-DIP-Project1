# file: evaluate.py

import os
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import argparse
import glob

def evaluate(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids):
    # --- 使用Faiss构建索引 ---
    # L2归一化后的特征，内积(IP)等价于余弦相似度
    index = faiss.IndexFlatIP(gallery_features.shape[1])
    index.add(gallery_features)
    # D: 距离/相似度, I: 索引
    D, I = index.search(query_features, k=len(gallery_pids))
    
    all_AP = []
    # CMC (Cumulative Matching Characteristics) 数组
    cmc = np.zeros(len(gallery_pids))
    
    for i in tqdm(range(len(query_pids)), desc="Evaluating"):
        query_pid = query_pids[i]
        query_camid = query_camids[i]
        
        # 获取当前查询的排序结果
        retrieved_indices = I[i]
        retrieved_pids = gallery_pids[retrieved_indices]
        retrieved_camids = gallery_camids[retrieved_indices]
        
        # --- 核心逻辑：识别 Good Match 和 Junk Match ---
        is_good_match = (retrieved_pids == query_pid) & (retrieved_camids != query_camid)
        is_junk_match = (retrieved_pids == query_pid) & (retrieved_camids == query_camid)
        
        # --- 创建一个mask来移除Junk Match ---
        mask_keep = ~is_junk_match
        
        # 如果gallery中没有任何一个good match，则跳过此查询
        if not np.any(is_good_match):
            continue
        
        # --- mAP 计算 ---
        # 过滤掉Junk Match后的排序列表
        final_ranked_good_matches = is_good_match[mask_keep]
        
        num_relevant = np.sum(final_ranked_good_matches)
        if num_relevant == 0:
            all_AP.append(0.0)
            continue
            
        cumulative_matches = np.cumsum(final_ranked_good_matches)
        precision_at_k = cumulative_matches / (np.arange(len(final_ranked_good_matches)) + 1)
        
        AP = np.sum(precision_at_k * final_ranked_good_matches) / num_relevant
        all_AP.append(AP)
        
        # --- CMC (Rank-k) 计算 ---
        # 在 *过滤掉junk* 的列表中，找到第一个正确匹配的位置
        # 注意：CMC通常是在原始列表中计算，但要跳过junk
        first_good_match_idx_in_filtered_list = np.where(final_ranked_good_matches)[0]
        if len(first_good_match_idx_in_filtered_list) > 0:
            # 排名从0开始，所以rank=index
            rank = first_good_match_idx_in_filtered_list[0]
            # 如果在rank `k` 找到了，那么对于所有 `j >= k` 的rank `j` 都算成功
            cmc[rank:] += 1

    # --- 计算最终指标 ---
    if len(all_AP) == 0:
        print("Warning: No valid queries found. mAP and Rank-1 will be 0.")
        return 0.0, 0.0

    mAP = np.mean(all_AP)
    # 将CMC计数转换为百分比
    cmc = cmc / len(all_AP)
    rank1 = cmc[0]
    
    return rank1, mAP

def load_features_from_parts(feats_dir, feats_prefix):
    """从多个部分文件中加载并合并特征。"""
    part_files = sorted(glob.glob(os.path.join(feats_dir, f"{feats_prefix}_part_*.pkl")))
    print(f"Found {len(part_files)} parts for prefix '{feats_prefix}'")
    
    if not part_files:
        raise FileNotFoundError(f"No feature files found for prefix '{feats_prefix}' in '{feats_dir}'")

    all_img_paths, all_pids, all_camids, feature_parts = [], [], [], []
    for part_file in part_files:
        with open(part_file, 'rb') as f:
            data = pickle.load(f)
        all_img_paths.extend(data['img_paths'])
        all_pids.extend(data['pids'])
        all_camids.extend(data['camids'])
        feature_parts.append(data['features'])
        
    all_features = np.vstack(feature_parts)
    
    return {
        'img_paths': all_img_paths,
        'pids': np.array(all_pids),
        'camids': np.array(all_camids),
        'features': all_features
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script for Re-ID")
    parser.add_argument('--feats_dir', type=str, required=True, help="Directory containing the feature part files")
    parser.add_argument('--query_prefix', type=str, required=True, help="Prefix for query feature files (e.g., 'baseline_query')")
    parser.add_argument('--gallery_prefix', type=str, required=True, help="Prefix for gallery feature files (e.g., 'baseline_gallery')")
    args = parser.parse_args()

    print("Loading query features...")
    query_data = load_features_from_parts(args.feats_dir, args.query_prefix)
    
    print("Loading gallery features...")
    gallery_data = load_features_from_parts(args.feats_dir, args.gallery_prefix)
    
    print(f"Query size: {len(query_data['pids'])}, Gallery size: {len(gallery_data['pids'])}")
    
    rank1, mAP = evaluate(
        query_data['features'], query_data['pids'], query_data['camids'],
        gallery_data['features'], gallery_data['pids'], gallery_data['camids']
    )
    
    print("="*20)
    print(f"Results for: {args.query_prefix.replace('_query', '')}")
    print(f"Rank-1: {rank1:.4f} ({rank1:.2%})")
    print(f"mAP:    {mAP:.4f} ({mAP:.2%})")
    print("="*20)