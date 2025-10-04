#!/usr/bin/env python3
"""
Secondary False Positive Filter
Small CNN classifier to reduce false positives from primary detector.
Uses ResNet18 backbone to classify candidate crops as polyp vs non-polyp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import numpy as np
from PIL import Image


class PolypClassifier(nn.Module):
    """
    Binary classifier for polyp vs non-polyp using ResNet18 backbone.
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize classifier.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final FC layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2)  # Binary: polyp vs non-polyp
        )
        
        print(f"✓ Initialized PolypClassifier with ResNet18 backbone")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Logits tensor (B, 2)
        """
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Probability tensor (B, 2)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classes and probabilities.
        
        Args:
            x: Input tensor (B, C, H, W)
            threshold: Classification threshold
        
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        probs = self.predict_proba(x)
        polyp_probs = probs[:, 1]  # Probability of being a polyp
        predictions = (polyp_probs >= threshold).long()
        return predictions, polyp_probs


class FalsePositiveFilter:
    """
    Wrapper for false positive filtering in inference pipeline.
    """
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None,
                 device: str = 'auto',
                 threshold: float = 0.5,
                 img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize FP filter.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use
            threshold: Classification threshold
            img_size: Input image size for classifier
        """
        self.device = self._get_device(device)
        self.threshold = threshold
        self.img_size = img_size
        
        # Initialize model
        self.model = PolypClassifier(pretrained=True)
        self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
    
    def _get_device(self, device: str) -> torch.device:
        """Determine device to use."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"✓ Loaded FP filter from {checkpoint_path}")
    
    def save_checkpoint(self, save_path: str, optimizer=None, epoch=None, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'img_size': self.img_size
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)
        print(f"✓ Saved FP filter to {save_path}")
    
    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess image crop for classification.
        
        Args:
            crop: Image crop as numpy array (H, W, C)
        
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Resize
        crop_pil = Image.fromarray(crop)
        crop_pil = crop_pil.resize(self.img_size, Image.BILINEAR)
        
        # Convert to tensor and normalize (ImageNet stats)
        crop_array = np.array(crop_pil).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        crop_array = (crop_array - mean) / std
        
        # Transpose to (C, H, W) and add batch dimension
        crop_tensor = torch.from_numpy(crop_array.transpose(2, 0, 1)).float()
        crop_tensor = crop_tensor.unsqueeze(0)
        
        return crop_tensor
    
    def filter_detections(self, 
                          image: np.ndarray,
                          bboxes: np.ndarray,
                          scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter detections by classifying each crop.
        
        Args:
            image: Full image as numpy array (H, W, C)
            bboxes: Bounding boxes (N, 4) in format [x1, y1, x2, y2]
            scores: Detection scores (N,)
        
        Returns:
            Filtered bboxes and scores
        """
        if len(bboxes) == 0:
            return bboxes, scores
        
        filtered_bboxes = []
        filtered_scores = []
        
        with torch.no_grad():
            for bbox, score in zip(bboxes, scores):
                # Extract crop
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                crop = image[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Preprocess and classify
                crop_tensor = self.preprocess_crop(crop).to(self.device)
                pred, prob = self.model.predict(crop_tensor, self.threshold)
                
                # Keep if classified as polyp
                if pred.item() == 1:
                    filtered_bboxes.append(bbox)
                    # Update score with classification probability
                    filtered_scores.append(score * prob.item())
        
        if len(filtered_bboxes) == 0:
            return np.array([]), np.array([])
        
        return np.array(filtered_bboxes), np.array(filtered_scores)
    
    def __call__(self, 
                 image: np.ndarray,
                 bboxes: np.ndarray,
                 scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter (shorthand for filter_detections).
        """
        return self.filter_detections(image, bboxes, scores)


def train_fp_filter(train_loader,
                   val_loader,
                   num_epochs: int = 50,
                   lr: float = 1e-4,
                   device: str = 'auto',
                   save_dir: str = 'logs/fp_filter'):
    """
    Train false positive filter.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
    
    Returns:
        Trained model
    """
    import os
    from pathlib import Path
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    fp_filter = FalsePositiveFilter(device=device)
    model = fp_filter.model
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(fp_filter.device)
            labels = labels.to(fp_filter.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(fp_filter.device)
                labels = labels.to(fp_filter.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_fp_filter.pt')
            fp_filter.save_checkpoint(save_path, optimizer, epoch, {'val_acc': val_acc})
        
        # Save last model
        save_path = os.path.join(save_dir, 'last_fp_filter.pt')
        fp_filter.save_checkpoint(save_path, optimizer, epoch, {'val_acc': val_acc})
    
    print(f"✓ Training complete. Best val acc: {best_val_acc:.2f}%")
    return fp_filter


# Testing
if __name__ == "__main__":
    print("Testing FP filter...")
    
    # Create model
    fp_filter = FalsePositiveFilter(device='cpu')
    
    # Test with dummy data
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    dummy_scores = np.array([0.8, 0.9])
    
    # Filter
    filtered_bboxes, filtered_scores = fp_filter(dummy_img, dummy_bboxes, dummy_scores)
    
    print(f"Input: {len(dummy_bboxes)} detections")
    print(f"Output: {len(filtered_bboxes)} detections")
    print("✓ FP filter test passed!")
