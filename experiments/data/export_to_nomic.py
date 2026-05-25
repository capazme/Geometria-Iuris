import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    import nomic
    from nomic import atlas
except ImportError:
    print("Error: Nomic is not installed.")
    print("Please install it by running:")
    print("    pip install nomic")
    print("\nThen login to your account (or create a free one) by running:")
    print("    nomic login")
    sys.exit(1)

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
    
    # Nomic requires a unique ID for each point. We'll add one.
    for i, item in enumerate(metadata):
        item["id"] = f"term_{i}"
    
    # Find all model directories
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "vectors.npy").exists()]
    model_dirs.sort()
    
    if not model_dirs:
        print("No embedding directories found!")
        return
        
    print("\nWhich model would you like to visualize in Nomic Atlas?")
    for i, model_dir in enumerate(model_dirs):
        print(f"  {i + 1}. {model_dir.name}")
        
    try:
        choice = input(f"Enter number (1-{len(model_dirs)}) [default 1]: ")
        if not choice.strip():
            choice = 1
        else:
            choice = int(choice)
        selected_model_dir = model_dirs[choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Selecting the first model by default...")
        selected_model_dir = model_dirs[0]
        
    model_name = selected_model_dir.name
    print(f"\nProceeding with model: {model_name}")
    
    vectors_path = selected_model_dir / "vectors.npy"
    print(f"Loading vectors from {vectors_path}...")
    vectors = np.load(vectors_path)
    
    print(f"Vectors shape: {vectors.shape}")
    
    if len(vectors) != len(metadata):
        print(f"Error: Metadata length ({len(metadata)}) does not match vectors length ({len(vectors)}).")
        return
        
    print(f"\n🚀 Creating Nomic Atlas project: THESIS-{model_name}...")
    print("This might take a minute or two depending on your internet connection.")
    print("If it's your first time, a browser window will pop up to show your map!")
    
    # Use topic_model=True to automatically cluster and name the legal topics
    project = atlas.map_data(
        embeddings=vectors,
        data=metadata,
        id_field="id",
        topic_model=True,
        name=f"THESIS-{model_name}",
        description=f"Interactive map of legal document embeddings using {model_name}"
    )
    
    print("\n" + "="*50)
    print("✅ UPLOAD COMPLETE")
    print("="*50)
    print(f"Your Atlas map should be open in your browser or available in your Nomic dashboard.")
    print("Enjoy exploring!")

if __name__ == "__main__":
    main()
