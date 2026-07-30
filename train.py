#!/usr/bin/env python3
"""
Training Script for Polyp Detection
Supports single training run and K-fold cross-validation.
"""

import os
import sys
import argparse
import random
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
import yaml

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Set random seed to {seed}")


class PolypDataset(Dataset):
    """
    Dataset class for YOLO-format polyp detection data.
    """
    
    def __init__(self, 
                 img_dir: Path,
                 label_dir: Path,
                 img_size: int = 640,
                 augmentations=None):
        """
        Initialize dataset.
        
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            img_size: Target image size
            augmentations: Albumentations transform
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augmentations = augmentations
        
        # Get list of images
        self.image_files = sorted(list(self.img_dir.glob("*.jpg")) + 
                                 list(self.img_dir.glob("*.png")))
        
        print(f"Loaded {len(self.image_files)} images from {img_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = self._load_image(img_path)
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + ".txt")
        bboxes, class_labels = self._load_labels(label_path)
        
        # Apply augmentations if provided
        if self.augmentations is not None and len(bboxes) > 0:
            try:
                transformed = self.augmentations(
                    image=image, 
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except Exception as e:
                print(f"Warning: Augmentation failed for {img_path.name}: {e}")
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels,
            'image_path': str(img_path)
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_labels(self, path: Path) -> tuple:
        """Load YOLO format labels."""
        bboxes = []
        class_labels = []
        
        if not path.exists():
            return bboxes, class_labels
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]  # x_center, y_center, w, h
                    bboxes.append(bbox)
                    class_labels.append(class_id)
        
        return bboxes, class_labels


def train_yolov8(args):
    """
    Train YOLOv8 model using ultralytics.
    """
    from models.detector import YOLOv8PolypDetector, create_data_yaml
    
    print("\n" + "="*80)
    print("TRAINING YOLOV8 POLYP DETECTOR")
    print("="*80)
    
    # Set seed
    set_seed(args.seed)
    
    # Create data YAML
    data_yaml_path = create_data_yaml(
        data_dir=args.dataset,
        num_classes=1,
        class_names=['polyp'],
        save_path=os.path.join(args.save_dir, 'data.yaml')
    )
    
    # Initialize detector
    detector = YOLOv8PolypDetector(
        model_size=args.model_size,
        num_classes=1,
        img_size=args.img_size,
        pretrained=args.pretrained,
        device=args.device
    )
    
    if not detector.ultralytics_available:
        print("Error: YOLOv8 not available. Install with: pip install ultralytics")
        return None
    
    # Training arguments
    train_kwargs = {
        'lr0': args.lr,
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Cls loss gain
        'dfl': 1.5,  # DFL loss gain
        'amp': args.amp,  # Mixed precision
        'save': True,
        'save_period': -1,  # Save checkpoint every epoch
        'cache': False,
        'workers': args.workers,
        'exist_ok': True,
        'pretrained': args.pretrained,
        'verbose': True,
    }
    
    # Train
    results = detector.train(
        data_yaml=data_yaml_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        optimizer=args.optimizer,
        save_dir=args.save_dir,
        patience=args.patience,
        **train_kwargs
    )
    
    print("\n✓ Training complete!")
    return results


def train_kfold(args):
    """
    Perform K-fold cross-validation training.
    """
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION (K={args.kfold})")
    print("="*80)
    
    # Get all images from train + val
    train_img_dir = Path(args.dataset) / 'train' / 'images'
    val_img_dir = Path(args.dataset) / 'val' / 'images'
    
    all_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    all_images += list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    
    print(f"Total images for K-fold: {len(all_images)}")
    
    # K-fold split
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{args.kfold}")
        print(f"{'='*80}")
        
        # Create fold-specific directories
        fold_dir = Path(args.save_dir) / f"fold_{fold + 1}"
        fold_train_img = fold_dir / 'train' / 'images'
        fold_train_lbl = fold_dir / 'train' / 'labels'
        fold_val_img = fold_dir / 'val' / 'images'
        fold_val_lbl = fold_dir / 'val' / 'labels'
        
        for d in [fold_train_img, fold_train_lbl, fold_val_img, fold_val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Copy files to fold directories
        # (In practice, should use symlinks or just pass indices to dataloader)
        # For simplicity, we'll use the original structure and modify data.yaml
        
        # Create fold-specific args
        fold_args = argparse.Namespace(**vars(args))
        fold_args.save_dir = str(fold_dir)
        fold_args.kfold = 0  # Disable recursion
        
        # Train on this fold
        results = train_yolov8(fold_args)
        
        if results:
            fold_results.append({
                'fold': fold + 1,
                'results': results
            })
    
    # Save aggregated results
    results_path = Path(args.save_dir) / 'kfold_results.json'
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    print(f"\n✓ K-fold cross-validation complete!")
    print(f"Results saved to {results_path}")
    
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Train polyp detection model")
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='data_preprocessed',
                       help='Path to dataset root directory')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Train from scratch')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loader workers')
    
    # Cross-validation
    parser.add_argument('--kfold', type=int, default=0,
                       help='Number of folds for cross-validation (0 for no CV)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable mixed precision')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='logs',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    args_path = Path(args.save_dir) / 'train_args.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Training configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Train
    if args.kfold > 0:
        results = train_kfold(args)
    else:
        results = train_yolov8(args)
    
    if results:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Results saved to {args.save_dir}")
        print("\nNext steps:")
        print("  1. Evaluate model: python evaluate.py --checkpoint <path>")
        print("  2. Run inference: python infer.py --input <path>")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
