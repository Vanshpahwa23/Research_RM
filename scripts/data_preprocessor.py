#!/usr/bin/env python3
"""
Data Preprocessor for Polyp Detection
Handles image loading, resizing, contrast enhancement, denoising, and video frame extraction.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import shutil


def load_image(path: Union[str, Path], as_rgb: bool = True) -> Optional[np.ndarray]:
    """
    Safely load an image with proper error handling.
    
    Args:
        path: Path to image file
        as_rgb: If True, convert to RGB; otherwise keep in BGR
    
    Returns:
        Image as numpy array or None if loading fails
    """
    try:
        path = str(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Warning: Failed to load image {path}")
            return None
        
        # Ensure proper dtype
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Convert BGR to RGB if requested
        if as_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def resize_image(img: np.ndarray, 
                 target_size: Tuple[int, int], 
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        img: Input image as numpy array
        target_size: Target (width, height)
        keep_aspect_ratio: If True, maintain aspect ratio and pad
    
    Returns:
        Resized image
    """
    if not keep_aspect_ratio:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Keep aspect ratio and pad
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def apply_clahe(img: np.ndarray, 
                clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        img: Input RGB image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # Merge channels
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return enhanced


def apply_gaussian_denoise(img: np.ndarray, 
                           kernel_size: int = 5, 
                           sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur for denoising.
    
    Args:
        img: Input image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Gaussian standard deviation
    
    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def convert_to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV color space."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def extract_frames_from_video(video_path: Union[str, Path],
                              output_dir: Union[str, Path],
                              fps: float = 2.0,
                              max_frames: Optional[int] = None) -> int:
    """
    Extract frames from video at specified fps.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract (None for all)
    
    Returns:
        Number of frames extracted
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    
    print(f"Video FPS: {video_fps}, Extracting every {frame_interval} frames")
    
    frame_count = 0
    extracted_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at interval
            if frame_count % frame_interval == 0:
                frame_filename = output_dir / f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path}")
    
    return extracted_count


def preprocess_image(img: np.ndarray,
                     target_size: Tuple[int, int] = (640, 480),
                     apply_clahe_enhance: bool = True,
                     apply_denoise: bool = False,
                     clahe_clip_limit: float = 2.0,
                     clahe_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply full preprocessing pipeline to an image.
    
    Args:
        img: Input image
        target_size: Target (width, height)
        apply_clahe_enhance: Whether to apply CLAHE
        apply_denoise: Whether to apply Gaussian denoising
        clahe_clip_limit: CLAHE clip limit
        clahe_grid_size: CLAHE grid size
    
    Returns:
        Preprocessed image
    """
    # Resize
    img = resize_image(img, target_size, keep_aspect_ratio=True)
    
    # Apply CLAHE
    if apply_clahe_enhance:
        img = apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_grid_size)
    
    # Apply denoising
    if apply_denoise:
        img = apply_gaussian_denoise(img)
    
    return img


def process_dataset(src_dir: Path,
                   dst_dir: Path,
                   target_size: Tuple[int, int] = (640, 480),
                   apply_clahe_enhance: bool = True,
                   apply_denoise: bool = False):
    """
    Process entire dataset (train/val/test splits).
    
    Args:
        src_dir: Source directory containing train/val/test subdirectories
        dst_dir: Destination directory for preprocessed data
        target_size: Target image size (width, height)
        apply_clahe_enhance: Whether to apply CLAHE
        apply_denoise: Whether to apply denoising
    """
    print(f"\nPreprocessing dataset from {src_dir} to {dst_dir}")
    print(f"Target size: {target_size}")
    print(f"CLAHE: {apply_clahe_enhance}, Denoise: {apply_denoise}")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        src_img_dir = src_dir / split / "images"
        src_label_dir = src_dir / split / "labels"
        dst_img_dir = dst_dir / split / "images"
        dst_label_dir = dst_dir / split / "labels"
        
        if not src_img_dir.exists():
            print(f"Warning: {src_img_dir} does not exist, skipping...")
            continue
        
        # Create destination directories
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        image_files = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
        
        print(f"\nProcessing {split} split: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Load image
            img = load_image(img_path, as_rgb=True)
            
            if img is None:
                continue
            
            # Preprocess
            img_processed = preprocess_image(
                img,
                target_size=target_size,
                apply_clahe_enhance=apply_clahe_enhance,
                apply_denoise=apply_denoise
            )
            
            # Save processed image
            dst_img_path = dst_img_dir / img_path.name
            img_bgr = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dst_img_path), img_bgr)
            
            # Copy corresponding label file if exists
            label_path = src_label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dst_label_path = dst_label_dir / (img_path.stem + ".txt")
                shutil.copy(label_path, dst_label_path)
        
        print(f"✓ Completed {split} split")
    
    print(f"\n✓ Dataset preprocessing complete!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess polyp detection dataset")
    parser.add_argument("--src", type=str, default="data",
                       help="Source directory containing raw data")
    parser.add_argument("--dst", type=str, default="data_preprocessed",
                       help="Destination directory for preprocessed data")
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 480],
                       help="Target image size (width height)")
    parser.add_argument("--clahe", action="store_true", default=True,
                       help="Apply CLAHE contrast enhancement")
    parser.add_argument("--no-clahe", action="store_false", dest="clahe",
                       help="Disable CLAHE")
    parser.add_argument("--denoise", action="store_true",
                       help="Apply Gaussian denoising")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0,
                       help="CLAHE clip limit")
    parser.add_argument("--clahe-grid-size", type=int, nargs=2, default=[8, 8],
                       help="CLAHE grid size")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    target_size = tuple(args.img_size)
    
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        return 1
    
    process_dataset(
        src_dir=src_dir,
        dst_dir=dst_dir,
        target_size=target_size,
        apply_clahe_enhance=args.clahe,
        apply_denoise=args.denoise
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
