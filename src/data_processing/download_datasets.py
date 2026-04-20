"""
Dataset Download Module
Downloads datasets from Kaggle (run once, reuse forever)
"""

import os
import sys
from pathlib import Path
import yaml


def download_datasets(config_path: str = "config.yaml"):
    """
    Download both detection and emotion datasets from Kaggle.
    Checks if data already exists before downloading.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    raw_data_dir = Path(config['paths']['raw_data'])
    detection_dir = raw_data_dir / "detection_dataset"
    emotion_dir = raw_data_dir / "emotion_dataset"
    
    detection_dir.mkdir(parents=True, exist_ok=True)
    emotion_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DATASET DOWNLOAD")
    print("=" * 80)
    
    # Download detection dataset
    print("\n[1/2] Checking Detection Dataset...")
    if _check_dataset_exists(detection_dir):
        print("✓ Detection dataset already exists. Skipping download.")
    else:
        print("Downloading detection dataset from Kaggle...")
        _download_kaggle_dataset(
            config['datasets']['detection']['kaggle_dataset'],
            str(detection_dir)
        )
    
    # Download emotion dataset
    print("\n[2/2] Checking Emotion Dataset...")
    if _check_dataset_exists(emotion_dir):
        print("✓ Emotion dataset already exists. Skipping download.")
    else:
        print("Downloading emotion dataset from Kaggle...")
        _download_kaggle_dataset(
            config['datasets']['emotion']['kaggle_dataset'],
            str(emotion_dir)
        )
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Detection dataset: {detection_dir}")
    print(f"Emotion dataset: {emotion_dir}")


def _check_dataset_exists(dataset_dir: Path) -> bool:
    """Check if dataset directory contains files."""
    if not dataset_dir.exists():
        return False
    
    # Check if directory has any files
    files = list(dataset_dir.rglob("*"))
    return len(files) > 0


def _download_kaggle_dataset(dataset_name: str, target_dir: str):
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'user/dataset-name')
        target_dir: Target directory for download
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Authenticate Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download and extract dataset
        print(f"  Downloading: {dataset_name}")
        api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        print(f"  ✓ Downloaded to: {target_dir}")
        
    except Exception as e:
        print(f"  ✗ Error downloading dataset: {e}")
        print("\nPlease ensure:")
        print("  1. You have installed kaggle: pip install kaggle")
        print("  2. You have set up Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("  3. You have accepted the dataset rules on Kaggle website")
        sys.exit(1)


if __name__ == "__main__":
    download_datasets()
