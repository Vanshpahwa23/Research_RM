#!/usr/bin/env python3
"""
Unit tests for model initialization.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.detector import (
    YOLOv8PolypDetector,
    FallbackDetector,
    create_detector,
    create_data_yaml
)


class TestModelInit:
    """Test suite for model initialization."""
    
    def test_fallback_detector_init(self):
        """Test fallback detector initialization."""
        model = FallbackDetector(num_classes=1, img_size=640)
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert model.num_classes == 1
        assert model.img_size == 640
    
    def test_fallback_detector_forward(self):
        """Test fallback detector forward pass."""
        model = FallbackDetector(num_classes=1, img_size=640)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 640, 640)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output is not None
        assert len(output.shape) == 4  # (B, C, H, W)
        assert output.shape[0] == batch_size
    
    def test_create_detector_fallback(self):
        """Test detector factory with fallback."""
        detector = create_detector(
            model_type='fallback',
            num_classes=1,
            img_size=640
        )
        
        assert detector is not None
        assert isinstance(detector, FallbackDetector)
    
    def test_yolov8_detector_init(self):
        """Test YOLOv8 detector initialization."""
        # This may fail if ultralytics not installed
        try:
            detector = YOLOv8PolypDetector(
                model_size='n',
                num_classes=1,
                img_size=640,
                pretrained=False,
                device='cpu'
            )
            
            if detector.ultralytics_available:
                assert detector.model is not None
                assert detector.device == 'cpu'
        except Exception as e:
            pytest.skip(f"YOLOv8 not available: {e}")
    
    def test_create_data_yaml(self):
        """Test data YAML creation."""
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Create YAML
            result_path = create_data_yaml(
                data_dir='data',
                num_classes=1,
                class_names=['polyp'],
                save_path=yaml_path
            )
            
            assert Path(result_path).exists()
            
            # Load and verify
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'path' in config
            assert 'train' in config
            assert 'val' in config
            assert 'nc' in config
            assert 'names' in config
            assert config['nc'] == 1
            assert config['names'] == ['polyp']
        
        finally:
            # Cleanup
            if Path(yaml_path).exists():
                Path(yaml_path).unlink()
    
    def test_detector_device_selection(self):
        """Test device selection logic."""
        detector = YOLOv8PolypDetector(device='cpu')
        assert detector.device == 'cpu'
        
        # Test auto device selection
        detector_auto = YOLOv8PolypDetector(device='auto')
        assert detector_auto.device in ['cpu', 'cuda']


class TestFalsePositiveFilter:
    """Test suite for false positive filter."""
    
    def test_fp_filter_init(self):
        """Test FP filter initialization."""
        from models.secondary_fp_filter import FalsePositiveFilter
        
        fp_filter = FalsePositiveFilter(device='cpu')
        
        assert fp_filter is not None
        assert fp_filter.model is not None
        assert fp_filter.device.type == 'cpu'
    
    def test_fp_filter_preprocess(self):
        """Test FP filter preprocessing."""
        from models.secondary_fp_filter import FalsePositiveFilter
        
        fp_filter = FalsePositiveFilter(device='cpu', img_size=(224, 224))
        
        # Create dummy crop
        crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Preprocess
        tensor = fp_filter.preprocess_crop(crop)
        
        assert tensor is not None
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_fp_filter_forward(self):
        """Test FP filter forward pass."""
        from models.secondary_fp_filter import FalsePositiveFilter
        
        fp_filter = FalsePositiveFilter(device='cpu')
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            logits = fp_filter.model(dummy_input)
        
        assert logits is not None
        assert logits.shape == (2, 2)  # Binary classification
    
    def test_fp_filter_predict(self):
        """Test FP filter prediction."""
        from models.secondary_fp_filter import FalsePositiveFilter
        
        fp_filter = FalsePositiveFilter(device='cpu', threshold=0.5)
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Predict
        with torch.no_grad():
            predictions, probabilities = fp_filter.model.predict(dummy_input)
        
        assert predictions is not None
        assert probabilities is not None
        assert len(predictions) == 2
        assert len(probabilities) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
