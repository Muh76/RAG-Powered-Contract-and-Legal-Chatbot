#!/usr/bin/env python3
"""
Download CUAD Dataset from Hugging Face
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_cuad_dataset():
    """Download CUAD dataset from Hugging Face"""
    
    print("📥 Downloading CUAD Dataset from Hugging Face...")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path("data/cuad")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the dataset
        print("🔄 Downloading CUAD dataset...")
        local_dir = snapshot_download(
            repo_id="Nadav-Timor/CUAD",
            local_dir=str(data_dir),
            repo_type="dataset"
        )
        
        print(f"✅ CUAD dataset downloaded successfully!")
        print(f"📁 Local directory: {local_dir}")
        
        # List downloaded files
        print(f"\n📋 Downloaded files:")
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  📄 {file_path.name}: {size_mb:.2f} MB")
        
        return str(data_dir)
        
    except Exception as e:
        print(f"❌ Error downloading CUAD: {e}")
        return None

if __name__ == "__main__":
    download_cuad_dataset()
