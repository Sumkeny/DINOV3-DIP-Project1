# evaluate.py
import numpy as np
import faiss
import os
import argparse
from tqdm import tqdm

def parse_filename_market1501(filename):
    """
    Parses a Market-1501 filename to get person ID and camera ID.
    Example: 0001_c1s1_001051_00.jpg -> (1, 1)
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    person_id = int(parts[0])
    camera_id = int(parts[1][1])
    return person_id, camera_id

def evaluate_reid(query_feats, query_paths, gallery_feats, gallery_paths, top_k=100):
    """
    Evaluates the Re-ID model using Rank-1 and mAP metrics.
    """
    # 1. Build Faiss index for the gallery
    # Inner product on L2-normalized features is equivalent to cosine similarity
    index = faiss.IndexFlatIP(gallery_feats.shape[1])
    index.add(gallery_feats)
    print(f"Faiss index built with {index.ntotal} gallery features.")

    # 2. Search for each query
    # D: distances (inner products), I: indices of nearest neighbors
    D, I = index.search(query_feats, k=top_k)

    # 3. Parse IDs and camera info
    query_pids = np.array([parse_filename_market1501(p)[0] for p in query_paths])
    query_cids = np.array([parse_filename_market1501(p)[1] for p in query_paths])
    gallery_pids = np.array([parse_filename_market1501(p)[0] for p in gallery_paths])
    gallery_cids = np.array([parse_filename_market1501(p)[1] for p in gallery_paths])
    
    # 4. Compute metrics
    cmc = np.zeros(len(gallery_pids))
    all_ap = []
    num_valid_queries = 0

    for i in tqdm(range(len(query_feats)), desc="Evaluating"):
        query_pid = query_pids[i]
        query_cid = query_cids[i]
        
        # Get ranked gallery indices for the current query
        ranked_indices = I[i]
        ranked_gallery_pids = gallery_pids[ranked_indices]
        ranked_gallery_cids = gallery_cids[ranked_indices]
        
        # Filter out junk images or the query image itself from the same camera
        # Market-1501 specific: junk images have pid -1 or 0. Query image itself has same pid and cid.
        valid_mask = ~((ranked_gallery_pids == query_pid) & (ranked_gallery_cids == query_cid))
        valid_mask &= (ranked_gallery_pids != -1) # Filter junk images
        
        # Apply the mask
        clean_gallery_pids = ranked_gallery_pids[valid_mask]
        
        # Ground truth matches in the entire (cleaned) gallery
        gt_matches = np.sum(gallery_pids[gallery_pids != -1] == query_pid)
        if gt_matches == 0:
            continue # Skip queries with no GT in gallery
        
        num_valid_queries += 1
        
        # --- Calculate CMC (Rank-k) ---
        matches_at_k = (clean_gallery_pids == query_pid)
        match_found = np.any(matches_at_k)
        if match_found:
            first_match_idx = np.where(matches_at_k)[0][0]
            cmc[first_match_idx:] += 1

        # --- Calculate AP (Average Precision) ---
        if not match_found:
            all_ap.append(0)
            continue
            
        hits = np.cumsum(matches_at_k)
        precision_at_k = hits / (np.arange(len(clean_gallery_pids)) + 1)
        ap = np.sum(precision_at_k * matches_at_k) / gt_matches
        all_ap.append(ap)

    # Finalize metrics
    cmc = cmc / num_valid_queries
    mAP = np.mean(all_ap)
    
    return cmc[0], mAP # Return Rank-1 and mAP

def main(args):
    # Load query data
    query_data = np.load(args.query_features)
    query_feats = query_data['features']
    query_paths = query_data['paths']
    
    # Load gallery data
    gallery_data = np.load(args.gallery_features)
    gallery_feats = gallery_data['features']
    gallery_paths = gallery_data['paths']
    
    print(f"Loaded {len(query_feats)} query features and {len(gallery_feats)} gallery features.")

    # Note: Ensure features are L2-normalized, which feature_extraction.py does.
    
    rank1, mAP = evaluate_reid(query_feats, query_paths, gallery_feats, gallery_paths)
    
    print("\n--- Evaluation Results ---")
    print(f"Rank-1 Accuracy: {rank1:.2%}")
    print(f"mAP            : {mAP:.2%}")
    print("--------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Re-ID performance.")
    parser.add_argument('--query_features', type=str, required=True, help="Path to the query features (.npz file).")
    parser.add_argument('--gallery_features', type=str, required=True, help="Path to the gallery features (.npz file).")
    
    args = parser.parse_args()
    main(args)

# Example command line usage:
# python evaluate.py --query_features features/market1501/query_feats.npz --gallery_features features/market1501/gallery_feats.npz