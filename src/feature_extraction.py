import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import pickle
import glob

from model import DINOv3ReID, AdapterHead

def parse_args():
    parser = argparse.ArgumentParser(description="Feature Extraction with Batched File Saving")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--subdir', type=str, required=True, choices=['query', 'bounding_box_test'], help="Subdirectory to process")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save feature files")
    parser.add_argument('--output_prefix', type=str, required=True, help="Prefix for the output pkl files (e.g., 'baseline_query')")
    parser.add_argument('--adapter_path', type=str, default=None, help="Path to trained adapter weights")
    parser.add_argument('--batch_save_size', type=int, default=4000, help="Batch size for saving to separate files. Lower this if still OOM.")
    return parser.parse_args()

def extract_and_save_features_final(model, adapter, data_dir, subdir, output_dir, output_prefix, batch_save_size, device):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_dir = os.path.join(data_dir, subdir)
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    # Clean up old part files before starting
    old_files = glob.glob(os.path.join(output_dir, f"{output_prefix}_part_*.pkl"))
    if old_files:
        print(f"Removing {len(old_files)} old part files...")
        for f in old_files: os.remove(f)

    batch_data = {'img_paths': [], 'pids': [], 'camids': [], 'features': []}
    part_num = 0
    
    print(f"Extracting features from {subdir} and saving in parts...")
    
    for i, img_name in enumerate(tqdm(image_files)):
        if img_name.startswith('-1'): continue
        
        img_path = os.path.join(img_dir, img_name)
        pid = int(img_name.split('_')[0])
        camid = int(img_name.split('_')[1][1])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            backbone_feat = model(img_tensor)
            if adapter:
                _, final_feat = adapter(backbone_feat)
                final_feat = torch.nn.functional.normalize(final_feat, dim=1)
            else:
                final_feat = backbone_feat
        
        batch_data['img_paths'].append(img_path)
        batch_data['pids'].append(pid)
        batch_data['camids'].append(camid)
        batch_data['features'].append(final_feat.cpu().numpy())
        
        if (i + 1) % batch_save_size == 0 or (i + 1) == len(image_files):
            batch_data['features'] = np.vstack(batch_data['features'])
            output_path = os.path.join(output_dir, f"{output_prefix}_part_{part_num}.pkl")
            
            with open(output_path, 'wb') as f:
                pickle.dump(batch_data, f)
            
            print(f"Saved part {part_num} to {output_path}")
            part_num += 1
            batch_data = {'img_paths': [], 'pids': [], 'camids': [], 'features': []}

    print(f"Finished extracting features for {subdir}.")

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DINOv3ReID().to(device)
    model.eval()
    adapter = None
    if args.adapter_path:
        adapter = AdapterHead().to(device)
        adapter.load_state_dict(torch.load(args.adapter_path))
        adapter.eval()
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    extract_and_save_features_final(model, adapter, args.data_dir, args.subdir, args.output_dir, args.output_prefix, args.batch_save_size, device)