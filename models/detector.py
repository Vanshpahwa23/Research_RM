#!/usr/bin/env python3
"""
Polyp Detector Model
Primary: YOLOv8 wrapper with ultralytics
Fallback: Simple detector stub for environments without ultralytics
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import yaml


class YOLOv8PolypDetector:
    """
    YOLOv8-based polyp detector using ultralytics.
    """
    
    def __init__(self, 
                 model_size: str = 'n',  # n, s, m, l, x
                 num_classes: int = 1,
                 img_size: int = 640,
                 pretrained: bool = True,
                 device: str = 'auto'):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            num_classes: Number of classes (1 for polyp)
            img_size: Input image size
            pretrained: Whether to load pretrained COCO weights
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = self._get_device(device)
        
        try:
            from ultralytics import YOLO
            self.ultralytics_available = True
            
            # Load pretrained model or create new
            if pretrained:
                model_name = f'yolov8{model_size}.pt'
                print(f"Loading pretrained YOLOv8{model_size} model...")
                self.model = YOLO(model_name)
            else:
                # Create model from scratch
                model_name = f'yolov8{model_size}.yaml'
                self.model = YOLO(model_name)
            
            print(f"✓ YOLOv8{model_size} loaded successfully on {self.device}")
            
        except ImportError:
            print("Warning: ultralytics not available. Using fallback detector.")
            self.ultralytics_available = False
            self.model = None
    
    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def train(self,
             data_yaml: str,
             epochs: int = 100,
             batch_size: int = 16,
             img_size: int = 640,
             lr0: float = 0.01,
             optimizer: str = 'AdamW',
             save_dir: str = 'logs',
             patience: int = 50,
             **kwargs) -> Dict:
        """
        Train the model.
        
        Args:
            data_yaml: Path to data YAML configuration
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            lr0: Initial learning rate
            optimizer: Optimizer type
            save_dir: Directory to save results
            patience: Early stopping patience
            **kwargs: Additional training arguments
        
        Returns:
            Training results dictionary
        """
        if not self.ultralytics_available or self.model is None:
            raise RuntimeError("YOLOv8 model not available")
        
        # Train
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr0,
            optimizer=optimizer,
            project=save_dir,
            name='train',
            patience=patience,
            device=self.device,
            **kwargs
        )
        
        return results
    
    def validate(self, data_yaml: str = None, **kwargs) -> Dict:
        """Validate the model."""
        if not self.ultralytics_available or self.model is None:
            raise RuntimeError("YOLOv8 model not available")
        
        results = self.model.val(data=data_yaml, **kwargs)
        return results
    
    def predict(self,
               source: Union[str, np.ndarray, Image.Image],
               conf: float = 0.25,
               iou: float = 0.45,
               **kwargs) -> List:
        """
        Run inference.
        
        Args:
            source: Image source (path, numpy array, or PIL Image)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            **kwargs: Additional prediction arguments
        
        Returns:
            List of prediction results
        """
        if not self.ultralytics_available or self.model is None:
            raise RuntimeError("YOLOv8 model not available")
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            **kwargs
        )
        
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """Export model to different formats."""
        if not self.ultralytics_available or self.model is None:
            raise RuntimeError("YOLOv8 model not available")
        
        return self.model.export(format=format, **kwargs)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if not self.ultralytics_available:
            raise RuntimeError("YOLOv8 not available")
        
        from ultralytics import YOLO
        self.model = YOLO(checkpoint_path)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # YOLOv8 saves automatically during training
        print(f"Model checkpoints saved during training to {save_path}")


class FallbackDetector(nn.Module):
    """
    Simple fallback detector for environments without ultralytics.
    This is a minimal stub for demonstration.
    """
    
    def __init__(self, num_classes: int = 1, img_size: int = 640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Simple CNN backbone (placeholder)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Detection head (simplified)
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5 + num_classes, kernel_size=1)  # x, y, w, h, conf, classes
        )
        
        print("Warning: Using fallback detector (not production-ready)")
        print("For production use, install ultralytics: pip install ultralytics")
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


def create_detector(model_type: str = 'yolov8',
                   model_size: str = 'n',
                   num_classes: int = 1,
                   img_size: int = 640,
                   pretrained: bool = True,
                   device: str = 'auto') -> Union[YOLOv8PolypDetector, FallbackDetector]:
    """
    Factory function to create detector.
    
    Args:
        model_type: Type of detector ('yolov8' or 'fallback')
        model_size: Model size for YOLOv8
        num_classes: Number of classes
        img_size: Input image size
        pretrained: Load pretrained weights
        device: Device to use
    
    Returns:
        Detector instance
    """
    if model_type.lower() == 'yolov8':
        detector = YOLOv8PolypDetector(
            model_size=model_size,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
            device=device
        )
        
        if not detector.ultralytics_available:
            print("Falling back to simple detector...")
            return FallbackDetector(num_classes=num_classes, img_size=img_size)
        
        return detector
    
    elif model_type.lower() == 'fallback':
        return FallbackDetector(num_classes=num_classes, img_size=img_size)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_data_yaml(data_dir: str,
                     num_classes: int = 1,
                     class_names: List[str] = None,
                     save_path: str = 'data.yaml') -> str:
    """
    Create YAML configuration file for YOLOv8 training.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        num_classes: Number of classes
        class_names: List of class names
        save_path: Path to save YAML file
    
    Returns:
        Path to saved YAML file
    """
    if class_names is None:
        class_names = ['polyp']
    
    data_dir = Path(data_dir).absolute()
    
    config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created data YAML at {save_path}")
    return save_path


# Testing
if __name__ == "__main__":
    print("Testing detector initialization...")
    
    # Try to create YOLOv8 detector
    detector = create_detector(
        model_type='yolov8',
        model_size='n',
        num_classes=1,
        pretrained=True
    )
    
    print(f"Detector type: {type(detector).__name__}")
    
    # Create dummy data YAML
    data_yaml_path = create_data_yaml(
        data_dir='data_preprocessed',
        num_classes=1,
        class_names=['polyp'],
        save_path='polyp_data.yaml'
    )
    
    print("✓ Detector initialization test passed!")
