from abc import ABC, abstractmethod

class BiasChecker(ABC):
    """
    Abstract base class for bias detection.
    """
    @abstractmethod
    def check(self, images, attributes=None):
        """
        Check for bias in the generated images.
        
        Args:
            images (torch.Tensor): Generated images.
            attributes (dict, optional): Attributes to check against (if conditioned).
            
        Returns:
            dict: A dictionary containing bias metrics/report.
        """
        pass

class Watermarker(ABC):
    """
    Abstract base class for watermarking.
    """
    @abstractmethod
    def apply(self, images):
        """
        Apply watermark to images.
        
        Args:
            images (torch.Tensor): Images to watermark.
            
        Returns:
            torch.Tensor: Watermarked images.
        """
        pass
    
    @abstractmethod
    def detect(self, images):
        """
        Detect watermark in images.
        
        Args:
            images (torch.Tensor): Images to check.
            
        Returns:
            bool: True if watermark detected, else False.
        """
        pass
