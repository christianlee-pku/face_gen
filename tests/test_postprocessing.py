import torch
import numpy as np
from src.apis.postprocessing import InvisibleWatermarker

def test_watermarker_tensor():
    watermarker = InvisibleWatermarker(input_shape=(3, 64, 64), signature_strength=0.1)
    
    # Create random image [-1, 1]
    img = torch.rand(2, 3, 64, 64) * 2 - 1
    
    # Apply
    watermarked = watermarker.apply(img)
    
    # Check shape
    assert watermarked.shape == img.shape
    # Check range
    assert watermarked.min() >= -1.0
    assert watermarked.max() <= 1.0
    
    # Check detection
    detected = watermarker.detect(watermarked)
    assert all(detected)
    
    # Check negative detection
    clean_img = torch.rand(2, 3, 64, 64) * 2 - 1
    detected_clean = watermarker.detect(clean_img)
    assert not any(detected_clean)

def test_watermarker_numpy():
    watermarker = InvisibleWatermarker(input_shape=(3, 64, 64), signature_strength=0.1)
    
    img = np.random.rand(2, 3, 64, 64).astype(np.float32) * 2 - 1
    
    watermarked = watermarker.apply(img)
    
    assert isinstance(watermarked, np.ndarray)
    assert watermarked.shape == img.shape
    
    detected = watermarker.detect(watermarked)
    assert all(detected)
