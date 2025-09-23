import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE Visualization")
    parser.add_argument('--features_path', type=str, required=True, help="Path to features file (.pkl)")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the plot")
    parser.add_argument('--num_pids', type=int, default=30, help="Number of PIDs to visualize")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading features from {args.features_path}")
    with open(args.features_path, 'rb') as f:
        data = pickle.load(f)

    pids = np.array(data['pids'])
    features = data['features']
    
    unique_pids = np.unique(pids)
    # Ensure we don't request more pids than available
    num_pids_to_sample = min(args.num_pids, len(unique_pids))
    selected_pids = np.random.choice(unique_pids, num_pids_to_sample, replace=False)
    
    mask = np.isin(pids, selected_pids)
    
    sample_features = features[mask]
    sample_pids = pids[mask]

    print(f"Running t-SNE on {len(sample_pids)} samples from {num_pids_to_sample} identities...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(sample_features)
    
    print("Plotting results...")
    plt.figure(figsize=(16, 12))
    cmap = plt.get_cmap('tab20', num_pids_to_sample)
    
    for i, pid in enumerate(tqdm(selected_pids, desc="Plotting PIDs")):
        pid_mask = (sample_pids == pid)
        plt.scatter(tsne_results[pid_mask, 0], tsne_results[pid_mask, 1], color=cmap(i), label=f'PID {pid}', s=10)

    plt.title(f't-SNE Visualization of Features ({os.path.basename(args.features_path)})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc="best", markerscale=2, fontsize='small')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path)
    print(f"Plot saved to {args.output_path}")

if __name__ == '__main__':
    main()