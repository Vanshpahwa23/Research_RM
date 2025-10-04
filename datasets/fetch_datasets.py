#!/usr/bin/env python3
"""
Dataset Fetcher for Polyp Detection
Attempts to download real datasets (Kvasir-SEG, CVC-ClinicDB, ETIS-Larib)
Falls back to synthetic dataset generation if downloads fail.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from tqdm import tqdm


def attempt_kaggle_download(dataset_id: str, dest_path: Path) -> bool:
    """Attempt to download dataset from Kaggle."""
    print(f"Attempting to download {dataset_id} from Kaggle...")
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest_path), "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ Successfully downloaded {dataset_id}")
            return True
        else:
            print(f"✗ Failed to download {dataset_id}: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Install with: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Download timeout for {dataset_id}")
        return False
    except Exception as e:
        print(f"✗ Error downloading {dataset_id}: {e}")
        return False


def attempt_wget_download(url: str, dest_path: Path, filename: str) -> bool:
    """Attempt to download dataset using wget."""
    print(f"Attempting to download from {url}...")
    try:
        cmd = ["wget", "-O", str(dest_path / filename), url, "--timeout=300"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✓ Successfully downloaded {filename}")
            return True
        else:
            print(f"✗ Failed to download {filename}: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ wget not found")
        return False
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False


def generate_synthetic_polyp(size: Tuple[int, int] = (640, 480), 
                              num_polyps: int = None) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Generate a synthetic colonoscopy image with polyp(s).
    
    Returns:
        image: RGB numpy array
        bboxes: List of bounding boxes in YOLO format [class_id, x_center, y_center, width, height]
    """
    width, height = size
    
    # Create realistic colonoscopy background
    # Simulate pinkish tissue color with texture
    base_color = np.array([220, 130, 120])  # Pinkish
    img = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16) + base_color
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Add texture using Gaussian noise and blur
    noise = np.random.normal(0, 15, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Apply bilateral filter for tissue-like appearance
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Add vascular patterns (darker lines)
    num_vessels = np.random.randint(3, 8)
    for _ in range(num_vessels):
        pts = []
        num_points = np.random.randint(3, 6)
        for _ in range(num_points):
            pts.append((np.random.randint(0, width), np.random.randint(0, height)))
        pts = np.array(pts, np.int32)
        color = (max(0, base_color[0] - 40), max(0, base_color[1] - 40), max(0, base_color[2] - 40))
        thickness = np.random.randint(1, 4)
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)
    
    # Add lighting gradient (vignette effect common in endoscopy)
    center_x, center_y = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1.0 - 0.3 * (dist_from_center / max_dist)
    img = (img * vignette[:, :, np.newaxis]).astype(np.uint8)
    
    # Generate polyps
    if num_polyps is None:
        num_polyps = np.random.randint(1, 3)  # 1-2 polyps per image
    
    bboxes = []
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    for _ in range(num_polyps):
        # Random polyp size (5-20% of image width)
        polyp_width = np.random.randint(int(width * 0.05), int(width * 0.2))
        polyp_height = np.random.randint(int(height * 0.05), int(height * 0.2))
        
        # Random position (avoid edges)
        margin = 50
        x = np.random.randint(margin, width - polyp_width - margin)
        y = np.random.randint(margin, height - polyp_height - margin)
        
        # Polyp shape (ellipse or irregular)
        if np.random.random() > 0.5:
            # Elliptical polyp
            bbox = [x, y, x + polyp_width, y + polyp_height]
            # Polyp color (slightly different from background)
            polyp_color = (
                min(255, base_color[0] + np.random.randint(-20, 40)),
                min(255, base_color[1] + np.random.randint(-20, 40)),
                min(255, base_color[2] + np.random.randint(-20, 40))
            )
            draw.ellipse(bbox, fill=polyp_color, outline=None)
            
            # Add texture to polyp
            for i in range(polyp_width // 3):
                spot_x = x + np.random.randint(0, polyp_width)
                spot_y = y + np.random.randint(0, polyp_height)
                spot_size = np.random.randint(2, 8)
                spot_color = (
                    max(0, polyp_color[0] - 20),
                    max(0, polyp_color[1] - 20),
                    max(0, polyp_color[2] - 20)
                )
                draw.ellipse([spot_x, spot_y, spot_x + spot_size, spot_y + spot_size], 
                            fill=spot_color)
        
        # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
        x_center = (x + polyp_width / 2) / width
        y_center = (y + polyp_height / 2) / height
        w_norm = polyp_width / width
        h_norm = polyp_height / height
        bboxes.append([0, x_center, y_center, w_norm, h_norm])  # class_id=0 for polyp
    
    img = np.array(img_pil)
    
    # Apply final blur and noise
    if np.random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Random brightness/contrast adjustment
    alpha = np.random.uniform(0.8, 1.2)  # contrast
    beta = np.random.randint(-20, 20)     # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    return img, bboxes


def generate_synthetic_dataset(dest_path: Path, 
                                num_train: int = 1000,
                                num_val: int = 200,
                                num_test: int = 200) -> Dict:
    """
    Generate synthetic polyp detection dataset.
    
    Args:
        dest_path: Destination directory
        num_train: Number of training images
        num_val: Number of validation images
        num_test: Number of test images
    
    Returns:
        Dataset manifest dictionary
    """
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC POLYP DATASET")
    print("="*80)
    print(f"Creating {num_train} train, {num_val} val, {num_test} test images...")
    
    manifest = {
        "source": "synthetic",
        "train": {"count": num_train, "images": [], "labels": []},
        "val": {"count": num_val, "images": [], "labels": []},
        "test": {"count": num_test, "images": [], "labels": []},
    }
    
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    
    for split_name, count in splits.items():
        print(f"\nGenerating {split_name} split ({count} images)...")
        
        img_dir = dest_path / split_name / "images"
        label_dir = dest_path / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(count), desc=f"Creating {split_name} images"):
            # Generate image
            img, bboxes = generate_synthetic_polyp(size=(640, 480))
            
            # Save image
            img_filename = f"{split_name}_{i:05d}.jpg"
            img_path = img_dir / img_filename
            Image.fromarray(img).save(img_path, quality=95)
            manifest[split_name]["images"].append(str(img_path))
            
            # Save YOLO format label
            label_filename = f"{split_name}_{i:05d}.txt"
            label_path = label_dir / label_filename
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # YOLO format: class x_center y_center width height
                    f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            manifest[split_name]["labels"].append(str(label_path))
    
    print(f"\n✓ Synthetic dataset created successfully!")
    print(f"  Train: {num_train} images")
    print(f"  Val: {num_val} images")
    print(f"  Test: {num_test} images")
    
    return manifest


def attempt_download_datasets(dest_path: Path) -> Tuple[bool, Dict]:
    """
    Attempt to download real polyp datasets from various sources.
    
    Returns:
        success: Whether any dataset was successfully downloaded
        manifest: Dataset information
    """
    print("\n" + "="*80)
    print("ATTEMPTING TO DOWNLOAD REAL POLYP DATASETS")
    print("="*80)
    
    manifest = {
        "source": "real",
        "datasets_downloaded": [],
        "download_attempts": []
    }
    
    # List of dataset sources to try
    datasets = [
        {
            "name": "Kvasir-SEG",
            "kaggle_id": "debeshjha1/kvasirseg",
            "urls": [
                "https://datasets.simula.no/downloads/kvasir-seg.zip"
            ]
        },
        {
            "name": "CVC-ClinicDB",
            "kaggle_id": "balraj98/cvcclinicdb",
            "urls": [
                "https://polyp.grand-challenge.org/CVCClinicDB/CVC-ClinicDB.zip"
            ]
        },
        {
            "name": "ETIS-Larib",
            "kaggle_id": "sani84/etis-larib-polyp-database",
            "urls": []
        }
    ]
    
    success = False
    
    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        print(f"\n--- Attempting {dataset_name} ---")
        
        # Try Kaggle
        if "kaggle_id" in dataset_info:
            attempt = {"method": "kaggle", "dataset": dataset_name, "success": False}
            if attempt_kaggle_download(dataset_info["kaggle_id"], dest_path):
                manifest["datasets_downloaded"].append(dataset_name)
                attempt["success"] = True
                success = True
            manifest["download_attempts"].append(attempt)
        
        # Try wget URLs
        for i, url in enumerate(dataset_info.get("urls", [])):
            attempt = {"method": "wget", "dataset": dataset_name, "url": url, "success": False}
            filename = f"{dataset_name}_{i}.zip"
            if attempt_wget_download(url, dest_path, filename):
                manifest["datasets_downloaded"].append(f"{dataset_name} (URL {i})")
                attempt["success"] = True
                success = True
            manifest["download_attempts"].append(attempt)
    
    return success, manifest


def main():
    parser = argparse.ArgumentParser(description="Fetch polyp detection datasets")
    parser.add_argument("--dest", type=str, default="data", 
                       help="Destination directory for datasets")
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Skip download attempts and generate synthetic data only")
    parser.add_argument("--num-train", type=int, default=1000,
                       help="Number of synthetic training images")
    parser.add_argument("--num-val", type=int, default=200,
                       help="Number of synthetic validation images")
    parser.add_argument("--num-test", type=int, default=200,
                       help="Number of synthetic test images")
    
    args = parser.parse_args()
    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    manifest = {}
    
    # Try to download real datasets first
    if not args.synthetic_only:
        success, download_manifest = attempt_download_datasets(dest_path)
        manifest.update(download_manifest)
        
        if success:
            print("\n✓ At least one real dataset was downloaded successfully!")
            print("Note: You may need to manually organize the downloaded files into YOLO format.")
            print("Expected structure:")
            print("  data/")
            print("    train/images/, train/labels/")
            print("    val/images/, val/labels/")
            print("    test/images/, test/labels/")
        else:
            print("\n✗ All download attempts failed. Falling back to synthetic dataset generation.")
    
    # Generate synthetic dataset (either as fallback or if requested)
    if args.synthetic_only or not manifest.get("datasets_downloaded"):
        synthetic_manifest = generate_synthetic_dataset(
            dest_path,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test
        )
        manifest.update(synthetic_manifest)
    
    # Save manifest
    manifest_path = dest_path / "dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Dataset manifest saved to {manifest_path}")
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    
    # Summary
    if manifest.get("source") == "synthetic":
        print(f"Generated synthetic dataset:")
        print(f"  Location: {dest_path}")
        print(f"  Train: {manifest['train']['count']} images")
        print(f"  Val: {manifest['val']['count']} images")
        print(f"  Test: {manifest['test']['count']} images")
    
    print("\nNext steps:")
    print("  1. Run preprocessing: python scripts/data_preprocessor.py")
    print("  2. Run training: python train.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
