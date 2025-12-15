import torch.nn as nn
from src.core.registry import MODELS

@MODELS.register_module()
class StyleGAN3Discriminator(nn.Module):
    """
    StyleGAN3 Discriminator.
    Often similar to StyleGAN2 discriminator but might include different
    augmentation pipelines or architecture tweaks for alias-free signal processing.
    """
    def __init__(self, img_resolution=256, img_channels=3):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Standard ResNet-like or VGG-like discriminator structure
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 16, 4, 2, 1), # 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), # 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), # 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), # 4
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1)
        )

    def forward(self, img, c=None, **kwargs):
        return self.main(img)