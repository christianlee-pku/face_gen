import torch
import numpy as np
from src.core.safeguards import Watermarker

class InvisibleWatermarker(Watermarker):
    """
    Implements an invisible watermark by adding a fixed noise signature.
    This signature can be detected by correlation analysis.
    """
    def __init__(self, input_shape=(3, 256, 256), signature_strength=0.02):
        """
        Args:
            input_shape (tuple): Shape of the image (C, H, W).
            signature_strength (float): Magnitude of the watermark noise.
        """
        self.signature_strength = signature_strength
        self.input_shape = input_shape
        
        # Generate a fixed random signature
        # Using a fixed seed ensures the same watermark is used/detected across instances
        generator = torch.Generator()
        generator.manual_seed(42)  # Fixed seed for the secret key
        self.signature = torch.randn(*input_shape, generator=generator)
        
        # Normalize signature to be binary (+strength or -strength) for robustness
        # This is a Spread Spectrum watermark variant
        self.signature = torch.sign(self.signature) * self.signature_strength

    def apply(self, images):
        """
        Apply watermark to images.
        
        Args:
            images (torch.Tensor or np.ndarray): Images in range [-1, 1].
            
        Returns:
            torch.Tensor or np.ndarray: Watermarked images.
        """
        is_numpy_input = isinstance(images, np.ndarray)
        if is_numpy_input:
            images_tensor = torch.from_numpy(images)
        else:
            images_tensor = images
        
        # Validate shape if possible (ignoring batch dim)
        if images_tensor.shape[1:] != self.signature.shape:
             # If shapes mismatch, we might need to resize signature or raise error
             # For now, let's assume correct shape or basic broadcasting
             pass

        # Ensure signature matches device and dtype
        if self.signature.device != images_tensor.device:
            self.signature = self.signature.to(images_tensor.device)
        if self.signature.dtype != images_tensor.dtype:
            self.signature = self.signature.to(dtype=images_tensor.dtype)
            
        # Add watermark (broadcasting over batch dimension)
        # images: (B, C, H, W), signature: (C, H, W)
        watermarked = images_tensor + self.signature.unsqueeze(0)
        
        # Clip to maintain valid range
        watermarked = torch.clamp(watermarked, -1.0, 1.0)
        
        if is_numpy_input:
            return watermarked.cpu().numpy()
        return watermarked

    def detect(self, images):
        """
        Detect watermark in images using correlation.
        
        Args:
            images (torch.Tensor or np.ndarray): Images to check.
            
        Returns:
            list[bool]: True if watermark detected for each image.
        """
        if isinstance(images, np.ndarray):
            images_tensor = torch.from_numpy(images)
        else:
            images_tensor = images
            
        # Ensure signature matches device and dtype
        if self.signature.device != images_tensor.device:
            self.signature = self.signature.to(images_tensor.device)
        if self.signature.dtype != images_tensor.dtype:
            self.signature = self.signature.to(dtype=images_tensor.dtype)
            
        # Calculate correlation per image in batch
        batch_size = images_tensor.shape[0]
        results = []
        
        # Detection metric: Mean(Image * Sign(Signature))
        # If watermarked: Mean((Content + Sig) * Sign(Sig)) 
        # = Mean(Content * Sign(Sig)) + Mean(Sig * Sign(Sig))
        # Content is uncorrelated -> 0
        # Sig * Sign(Sig) = |Sig| = strength
        # So metric should be approx `strength`.
        
        # Threshold: we check if metric is closer to strength than to 0.
        # Let's say > strength/2.
        
        sign_signature = torch.sign(self.signature)
        
        for i in range(batch_size):
            metric = torch.mean(images_tensor[i] * sign_signature)
            results.append(metric.item() > (self.signature_strength * 0.5))
            
        return results
