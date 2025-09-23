# src/evaluate.py (已修改以适应 Colab)
import numpy as np
import faiss
import os
import argparse
from tqdm import tqdm

def parse_filename_market1501(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    # 处理 'junk' or 'distractors' 图片, 它们的 person ID 为 -1 或 0
    if parts[0] in ['-1', '0000']:
        return -1, -1
    person_id = int(parts[0])
    camera_id = int(parts[1][1])
    return person_id, camera_id

def evaluate_reid(query_feats, query_paths, gallery_feats, gallery_paths, top_k=100):
    index = faiss.IndexFlatIP(gallery_feats.shape[1])
    index.add(gallery_feats)
    print(f"Faiss index built with {index.ntotal} gallery features.")

    D, I = index.search(query_feats, k=top_k)

    query_pids = np.array([parse_filename_market1501(p)[0] for p in query_paths])
    query_cids = np.array([parse_filename_market1501(p)[1] for p in query_paths])
    gallery_pids = np.array([parse_filename_market1501(p)[0] for p in gallery_paths])
    gallery_cids = np.array([parse_filename_market1501(p)[1] for p in gallery_paths])
    
    cmc = np.zeros(len(gallery_pids))
    all_ap = []
    num_valid_queries = 0

    for i in tqdm(range(len(query_feats)), desc="Evaluating"):
        query_pid = query_pids[i]
        query_cid = query_cids[i]
        
        ranked_indices = I[i]
        ranked_gallery_pids = gallery_pids[ranked_indices]
        ranked_gallery_cids = gallery_cids[ranked_indices]
        
        valid_mask = ~((ranked_gallery_pids == query_pid) & (ranked_gallery_cids == query_cid))
        valid_mask &= (ranked_gallery_pids != -1)
        
        clean_gallery_pids = ranked_gallery_pids[valid_mask]
        
        # 修正：gallery_pids[valid_gallery_mask]
        valid_gallery_mask = (gallery_pids != -1)
        gt_matches = np.sum(gallery_pids[valid_gallery_mask] == query_pid)
        if gt_matches == 0:
            continue
        
        num_valid_queries += 1
        
        matches_at_k = (clean_gallery_pids == query_pid)
        match_found = np.any(matches_at_k)

        if match_found:
            first_match_idx = np.where(matches_at_k)[0][0]
            cmc[first_match_idx:] += 1

        if not match_found:
            all_ap.append(0)
            continue
            
        hits = np.cumsum(matches_at_k)
        precision_at_k = hits / (np.arange(len(clean_gallery_pids)) + 1)
        ap = np.sum(precision_at_k * matches_at_k) / gt_matches
        all_ap.append(ap)

    cmc = cmc / num_valid_queries
    mAP = np.mean(all_ap)
    
    return cmc[0], mAP

def main(args):
    # --- 关键修改：从 args 组合路径 ---
    query_path = os.path.join(args.feats_dir, f"{args.query_prefix}_feats.npz")
    gallery_path = os.path.join(args.feats_dir, f"{args.gallery_prefix}_feats.npz")
    # ----------------------------------

    query_data = np.load(query_path)
    query_feats = query_data['features']
    query_paths = query_data['paths']
    
    gallery_data = np.load(gallery_path)
    gallery_feats = gallery_data['features']
    gallery_paths = gallery_data['paths']
    
    print(f"Loaded {len(query_feats)} query features from {query_path}")
    print(f"Loaded {len(gallery_feats)} gallery features from {gallery_path}")

    rank1, mAP = evaluate_reid(query_feats, query_paths, gallery_feats, gallery_paths)
    
    print("\n" + "---" * 10)
    print(" Baseline Evaluation Results on Market-1501")
    print("---" * 10)
    print(f"  Rank-1 Accuracy: {rank1:.2%}")
    print(f"  mAP            : {mAP:.2%}")
    print("---" * 10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Re-ID performance.")
    # --- 关键修改：更新命令行参数 ---
    parser.add_argument('--feats_dir', type=str, required=True, help="Directory where feature files are stored.")
    parser.add_argument('--query_prefix', type=str, required=True, help="Prefix of the query features file.")
    parser.add_argument('--gallery_prefix', type=str, required=True, help="Prefix of the gallery features file.")
    
    args = parser.parse_args()
    main(args)