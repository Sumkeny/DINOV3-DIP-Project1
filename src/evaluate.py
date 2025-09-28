import numpy as np
import faiss
from tqdm import tqdm
import argparse
import os

from config import *

def evaluate(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids, top_k=TOP_K):
    # 构建 Faiss 索引
    index = faiss.IndexFlatIP(gallery_features.shape[1])
    index.add(gallery_features)

    # 搜索
    D, I = index.search(query_features, k=top_k)

    # 计算 CMC (Cumulative Matching Characteristics) for Rank-k
    cmc = np.zeros(top_k)
    # 计算 AP (Average Precision)
    aps = []

    for i in tqdm(range(len(query_pids)), desc="Evaluating"):
        query_pid = query_pids[i]
        query_camid = query_camids[i]
        
        # 检索到的 gallery 索引
        gallery_indices = I[i]
        
        # 检索到的 pid 和 camid
        retrieved_pids = gallery_pids[gallery_indices]
        retrieved_camids = gallery_camids[gallery_indices]
        
        # 移除同一摄像头下的同一ID (Re-ID 评估标准)
        valid_indices = (retrieved_pids != query_pid) | (retrieved_camids != query_camid)
        
        # 筛选出有效的匹配结果
        matches = (retrieved_pids[valid_indices] == query_pid)
        
        if not np.any(matches):
            aps.append(0)
            continue
            
        # CMC
        cmc[np.where(matches)[0][0]:] += 1

        # mAP
        num_relevant = np.sum(matches)
        precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = np.sum(precision_at_k * matches) / num_relevant
        aps.append(ap)
        
    cmc /= len(query_pids)
    mAP = np.mean(aps)
    
    return cmc, mAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--adapted', action='store_true', help='Use features from adapted model')
    args = parser.parse_args()

    # 加载特征
    feature_dir = os.path.join(OUTPUT_DIR, 'features', args.dataset)
    suffix = '_adapted.npz' if args.adapted else '.npz'
    
    query_data = np.load(os.path.join(feature_dir, 'query_features' + suffix))
    gallery_data = np.load(os.path.join(feature_dir, 'gallery_features' + suffix))
    
    query_features, query_pids, query_camids = query_data['features'], query_data['pids'], query_data['camids']
    gallery_features, gallery_pids, gallery_camids = gallery_data['features'], gallery_data['pids'], gallery_data['camids']
    
    # 评估
    cmc, mAP = evaluate(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids)

    print(f"--- Results for {args.dataset} {'(Adapted)' if args.adapted else '(Baseline)'} ---")
    print(f"mAP: {mAP:.4f}")
    print(f"Rank-1: {cmc[0]:.4f}")
    print(f"Rank-5: {cmc[4]:.4f}")
    print(f"Rank-10: {cmc[9]:.4f}")