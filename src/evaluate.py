# file: src/evaluate.py
import os
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import argparse

def evaluate(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids):
    # ... (这个函数的核心逻辑完全不变) ...
    index = faiss.IndexFlatIP(gallery_features.shape[1])
    index.add(gallery_features)
    D, I = index.search(query_features, k=len(gallery_pids))
    all_AP = []
    cmc = np.zeros(len(gallery_pids))
    for i in tqdm(range(len(query_pids)), desc="Evaluating"):
        query_pid = query_pids[i]
        query_camid = query_camids[i]
        retrieved_indices = I[i]
        retrieved_pids = gallery_pids[retrieved_indices]
        retrieved_camids = gallery_camids[retrieved_indices]
        is_good_match = (retrieved_pids == query_pid) & (retrieved_camids != query_camid)
        is_junk_match = (retrieved_pids == query_pid) & (retrieved_camids == query_camid)
        mask_keep = ~is_junk_match
        if not np.any(is_good_match):
            continue
        final_ranked_good_matches = is_good_match[mask_keep]
        num_relevant = np.sum(final_ranked_good_matches)
        if num_relevant == 0:
            all_AP.append(0.0)
            continue
        cumulative_matches = np.cumsum(final_ranked_good_matches)
        precision_at_k = cumulative_matches / (np.arange(len(final_ranked_good_matches)) + 1)
        AP = np.sum(precision_at_k * final_ranked_good_matches) / num_relevant
        all_AP.append(AP)
        first_good_match_idx_in_filtered_list = np.where(final_ranked_good_matches)[0]
        if len(first_good_match_idx_in_filtered_list) > 0:
            rank = first_good_match_idx_in_filtered_list[0]
            cmc[rank:] += 1
    if len(all_AP) == 0:
        return 0.0, 0.0
    mAP = np.mean(all_AP)
    cmc = cmc / len(all_AP)
    rank1 = cmc[0]
    return rank1, mAP

# --- 这是唯一的修改点 ---
def load_features(feats_path):
    """直接从单一的 .pkl 文件加载特征。"""
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Feature file not found at: {feats_path}")
    with open(feats_path, 'rb') as f:
        data = pickle.load(f)
    # 确保 pids 和 camids 是 numpy 数组
    data['pids'] = np.array(data['pids'])
    data['camids'] = np.array(data['camids'])
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script for Re-ID")
    parser.add_argument('--feats_dir', type=str, required=True, help="Directory containing the feature files")
    parser.add_argument('--query_prefix', type=str, required=True, help="Prefix for the query feature file")
    parser.add_argument('--gallery_prefix', type=str, required=True, help="Prefix for the gallery feature file")
    args = parser.parse_args()

    query_path = os.path.join(args.feats_dir, f"{args.query_prefix}.pkl")
    gallery_path = os.path.join(args.feats_dir, f"{args.gallery_prefix}.pkl")

    print(f"Loading query features from {query_path}...")
    query_data = load_features(query_path)
    
    print(f"Loading gallery features from {gallery_path}...")
    gallery_data = load_features(gallery_path)
    
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