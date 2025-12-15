import torch
import os
import sys

# Add project root to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.generator import StyleGAN3Generator

def create_dummy_checkpoint(path, z_dim=512, w_dim=512, img_resolution=256, img_channels=3):
    """
    Creates a dummy StyleGAN3Generator checkpoint.
    """
    print(f"Creating dummy checkpoint at {path}...")
    generator = StyleGAN3Generator(
        z_dim=z_dim,
        w_dim=w_dim,
        img_resolution=img_resolution,
        img_channels=img_channels
    )
    # Just save the state_dict of the generator
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(generator.state_dict(), path)
    print("Dummy checkpoint created.")

if __name__ == "__main__":
    output_path = "work_dirs/stylegan3_celeba/epoch_5.pth"
    create_dummy_checkpoint(output_path)
