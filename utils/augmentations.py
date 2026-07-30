#!/usr/bin/env python3
"""
Data Augmentation for Polyp Detection
Uses albumentations library for robust, bbox-aware augmentations.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional


def get_training_augmentations(img_size: Tuple[int, int] = (640, 480),
                               p: float = 0.5) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Args:
        img_size: Target image size (width, height)
        p: Probability of applying augmentations
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        
        # Spatial transformations that preserve bboxes
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        # Elastic transform for realistic deformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=0,
            p=0.3
        ),
        
        # Optical distortion (common in endoscopy)
        A.OpticalDistortion(
            distort_limit=0.3,
            shift_limit=0.3,
            border_mode=0,
            p=0.3
        ),
        
        # Color/intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Color jitter
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=15,
            p=0.3
        ),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # CLAHE
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # Shadow and lighting
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.3
        ),
        
        # Random crop and resize (preserve bboxes)
        A.RandomSizedBBoxSafeCrop(
            height=img_size[1],
            width=img_size[0],
            erosion_rate=0.2,
            p=0.4
        ),
        
        # Final resize to ensure consistent size
        A.Resize(height=img_size[1], width=img_size[0], p=1.0),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
        min_area=100
    ))


def get_validation_augmentations(img_size: Tuple[int, int] = (640, 480)) -> A.Compose:
    """
    Get augmentation pipeline for validation (minimal augmentations).
    
    Args:
        img_size: Target image size (width, height)
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0], p=1.0),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
    ))


def get_test_time_augmentations(img_size: Tuple[int, int] = (640, 480)) -> List[A.Compose]:
    """
    Get list of test-time augmentation pipelines.
    Results can be averaged for more robust predictions.
    
    Args:
        img_size: Target image size (width, height)
    
    Returns:
        List of Albumentations Compose objects
    """
    base_transform = [A.Resize(height=img_size[1], width=img_size[0], p=1.0)]
    
    augmentations = [
        # Original
        A.Compose(base_transform, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        # Horizontal flip
        A.Compose(base_transform + [A.HorizontalFlip(p=1.0)], 
                 bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        # Vertical flip
        A.Compose(base_transform + [A.VerticalFlip(p=1.0)],
                 bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        # Brightness adjustment
        A.Compose(base_transform + [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)],
                 bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    ]
    
    return augmentations


def augment_image_and_bboxes(image: np.ndarray,
                             bboxes: List[List[float]],
                             class_labels: List[int],
                             transform: A.Compose) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Apply augmentation to image and bounding boxes.
    
    Args:
        image: Input image (H, W, C)
        bboxes: List of bboxes in YOLO format [x_center, y_center, width, height]
        class_labels: List of class labels for each bbox
        transform: Albumentations transform
    
    Returns:
        Augmented image, bboxes, and class labels
    """
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    aug_image = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_labels = transformed['class_labels']
    
    return aug_image, aug_bboxes, aug_labels


def convert_bbox_yolo_to_pascal(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Convert YOLO format bbox to Pascal VOC format.
    
    Args:
        bbox: [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x_min, y_min, x_max, y_max] in pixels
    """
    x_center, y_center, width, height = bbox
    
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return [x_min, y_min, x_max, y_max]


def convert_bbox_pascal_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    """
    Convert Pascal VOC format bbox to YOLO format.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x_center, y_center, width, height] (normalized 0-1)
    """
    x_min, y_min, x_max, y_max = bbox
    
    width_px = x_max - x_min
    height_px = y_max - y_min
    x_center_px = x_min + width_px / 2
    y_center_px = y_min + height_px / 2
    
    x_center = x_center_px / img_width
    y_center = y_center_px / img_height
    width = width_px / img_width
    height = height_px / img_height
    
    return [x_center, y_center, width, height]


def visualize_augmentation(image: np.ndarray,
                          bboxes: List[List[float]],
                          class_labels: List[int],
                          transform: A.Compose,
                          num_examples: int = 5) -> List[Tuple[np.ndarray, List[List[float]]]]:
    """
    Generate multiple augmented examples for visualization.
    
    Args:
        image: Input image
        bboxes: Bounding boxes in YOLO format
        class_labels: Class labels
        transform: Augmentation transform
        num_examples: Number of examples to generate
    
    Returns:
        List of (augmented_image, augmented_bboxes) tuples
    """
    examples = []
    
    for _ in range(num_examples):
        aug_img, aug_bboxes, _ = augment_image_and_bboxes(
            image, bboxes, class_labels, transform
        )
        examples.append((aug_img, aug_bboxes))
    
    return examples


# Example usage and testing
if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing augmentation pipelines...")
    
    # Create dummy image and bbox
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bboxes = [[0.5, 0.5, 0.2, 0.2]]  # Single bbox at center
    dummy_labels = [0]
    
    # Get transforms
    train_transform = get_training_augmentations((640, 480))
    val_transform = get_validation_augmentations((640, 480))
    
    # Apply augmentation
    aug_img, aug_bboxes, aug_labels = augment_image_and_bboxes(
        dummy_img, dummy_bboxes, dummy_labels, train_transform
    )
    
    print(f"Original image shape: {dummy_img.shape}")
    print(f"Augmented image shape: {aug_img.shape}")
    print(f"Original bbox: {dummy_bboxes}")
    print(f"Augmented bbox: {aug_bboxes}")
    print("✓ Augmentation test passed!")
