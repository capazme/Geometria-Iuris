import os
import json
import numpy as np
from pathlib import Path
import csv

def main():
    base_dir = Path("/Users/gpuzio/Desktop/CODE/THESIS/experiments/data/processed/embeddings")
    index_path = base_dir / "index.json"
    
    if not index_path.exists():
        print(f"Error: Could not find {index_path}")
        return

    print(f"Loading metadata from {index_path}...")
    with open(index_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    print(f"Loaded {len(metadata)} items.")
    
    headers = ["en", "zh_canonical", "domain", "tier"]
    
    # TensorBoard logs directory
    tb_logs_dir = base_dir / "tensorboard_logs"
    tb_logs_dir.mkdir(exist_ok=True)
    
    # Find all model directories
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "vectors.npy").exists()]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing model: {model_name}")
        
        vectors_path = model_dir / "vectors.npy"
        vectors = np.load(vectors_path)
        
        print(f"Loaded vectors with shape: {vectors.shape}")
        
        if len(vectors) != len(metadata):
            print(f"WARNING: Metadata length ({len(metadata)}) does not match vectors length ({len(vectors)}). Skipping...")
            continue
            
        model_log_dir = tb_logs_dir / model_name
        model_log_dir.mkdir(exist_ok=True)
        
        metadata_tsv_path = model_log_dir / "metadata.tsv"
        tensors_tsv_path = model_log_dir / "tensors.tsv"
        
        # Write metadata.tsv
        print(f"  -> Writing {metadata_tsv_path}...")
        with open(metadata_tsv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(headers)
            for item in metadata:
                row = [item.get(h, "") for h in headers]
                writer.writerow(row)
                
        # Write tensors.tsv
        print(f"  -> Writing {tensors_tsv_path}...")
        with open(tensors_tsv_path, 'w', encoding='utf-8') as f:
            for vec in vectors:
                f.write('\t'.join(map(str, vec)) + '\n')
                
    print("\n" + "="*50)
    print("✅ GENERATION COMPLETE")
    print("="*50)
    print("\nTo visualize in TensorBoard, follow these steps:")
    print("1. Open a new terminal")
    print(f"2. Run: tensorboard --logdir=\"{tb_logs_dir}\"")
    print("   (Note: you might need to run 'pip install tensorboard' first)")
    print("3. Open your browser and go to http://localhost:6006")
    print("4. In the top-right corner dropdown menu, select 'PROJECTOR' (or just go to http://localhost:6006/#projector)")
    print("5. On the left panel, click 'Load'")
    print("6. Upload 'tensors.tsv' as Step 1 and 'metadata.tsv' as Step 2 from any model folder you want to visualize.")
    print("7. Enjoy exploring! You can search terms, color by domain, or toggle UMAP/T-SNE/PCA.")

if __name__ == "__main__":
    main()
