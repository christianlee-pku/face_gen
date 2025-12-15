import torch
from src.core.registry import MODELS
from src.core.config import Config
import argparse
import os

# Register modules
import src.datasets.celeba
import src.models.generator
import src.models.discriminator
import src.models.losses

def export_onnx(config_path, checkpoint_path, output_path):
    """
    Export StyleGAN3 Generator to ONNX.
    """
    # 1. Load Config
    print(f"Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)
    
    # 2. Build Model
    print("Building model...")
    G = MODELS.build(cfg.model.generator)
    
    # 3. Load Weights
    print(f"Loading checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle both full checkpoint (with 'G') and weight-only
    state_dict = checkpoint['G'] if 'G' in checkpoint else checkpoint
    G.load_state_dict(state_dict)
    G.eval()
    
    # 4. Dummy Input
    # StyleGAN3 input is Z (latent)
    z_dim = cfg.model.generator.z_dim
    z = torch.randn(1, z_dim)
    
    # 5. Export
    print(f"Exporting to {output_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    torch.onnx.export(
        G,
        (z,),
        output_path,
        input_names=['z'],
        output_names=['image'],
        opset_version=18,
        # Allow variable batch size
        dynamic_axes={'z': {0: 'batch_size'}, 'image': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("checkpoint", help="Path to checkpoint .pth file")
    parser.add_argument("--out", help="Output .onnx file", default="work_dirs/model.onnx")
    args = parser.parse_args()
    
    export_onnx(args.config, args.checkpoint, args.out)
