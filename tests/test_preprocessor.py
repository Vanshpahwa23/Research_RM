#!/usr/bin/env python3
"""
Unit tests for data preprocessor.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_preprocessor import (
    load_image,
    resize_image,
    apply_clahe,
    apply_gaussian_denoise,
    preprocess_image
)


class TestPreprocessor:
    """Test suite for preprocessor functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create dummy image
        self.test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_img_path = self.temp_path / "test_image.jpg"
        cv2.imwrite(str(self.test_img_path), self.test_img)
    
    def teardown_method(self):
        """Cleanup after tests."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_image(self):
        """Test image loading."""
        # Test successful load
        img = load_image(self.test_img_path, as_rgb=True)
        assert img is not None
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8
        
        # Test RGB conversion
        img_rgb = load_image(self.test_img_path, as_rgb=True)
        img_bgr = load_image(self.test_img_path, as_rgb=False)
        assert not np.array_equal(img_rgb, img_bgr)  # Should be different
        
        # Test invalid path
        img_none = load_image(self.temp_path / "nonexistent.jpg")
        assert img_none is None
    
    def test_resize_image(self):
        """Test image resizing."""
        img = self.test_img
        
        # Test exact resize (no aspect ratio)
        resized = resize_image(img, (320, 240), keep_aspect_ratio=False)
        assert resized.shape == (240, 320, 3)
        
        # Test aspect ratio preserve
        resized_ar = resize_image(img, (320, 240), keep_aspect_ratio=True)
        assert resized_ar.shape == (240, 320, 3)
    
    def test_apply_clahe(self):
        """Test CLAHE enhancement."""
        # Convert to RGB
        img_rgb = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        enhanced = apply_clahe(img_rgb)
        
        assert enhanced.shape == img_rgb.shape
        assert enhanced.dtype == np.uint8
        assert not np.array_equal(enhanced, img_rgb)  # Should be different
    
    def test_apply_gaussian_denoise(self):
        """Test Gaussian denoising."""
        denoised = apply_gaussian_denoise(self.test_img, kernel_size=5)
        
        assert denoised.shape == self.test_img.shape
        assert denoised.dtype == np.uint8
    
    def test_preprocess_image(self):
        """Test full preprocessing pipeline."""
        img_rgb = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2RGB)
        
        # Test with CLAHE
        processed = preprocess_image(
            img_rgb,
            target_size=(640, 480),
            apply_clahe_enhance=True,
            apply_denoise=False
        )
        
        assert processed.shape == (480, 640, 3)
        assert processed.dtype == np.uint8
        
        # Test with denoising
        processed_denoised = preprocess_image(
            img_rgb,
            target_size=(640, 480),
            apply_clahe_enhance=True,
            apply_denoise=True
        )
        
        assert processed_denoised.shape == (480, 640, 3)
        assert processed_denoised.dtype == np.uint8
    
    def test_resize_aspect_ratio(self):
        """Test that aspect ratio is preserved correctly."""
        # Create rectangular image
        rect_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Resize with aspect ratio
        resized = resize_image(rect_img, (640, 480), keep_aspect_ratio=True)
        
        # Check output size
        assert resized.shape == (480, 640, 3)
        
        # The resized content should not fill the entire image (padding should exist)
        # We can check if there are black borders
        # This is a basic check - in practice would need more sophisticated verification


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
