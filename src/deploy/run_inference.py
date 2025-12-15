import argparse
import numpy as np
import torch
import os
from torchvision.utils import save_image
from src.core.registry import MODELS
from src.core.config import Config
import src.models.generator # Register generator

def run_inference(config_path, checkpoint_path, seeds, truncation, outdir):
    """
    Run inference using PyTorch model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config
    cfg = Config.fromfile(config_path)
    
    # 2. Build Model
    print("Building model...")
    G = MODELS.build(cfg.model.generator).to(device)
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle state dict key mismatch if necessary (e.g. if saved under 'G')
    state_dict = checkpoint['G'] if 'G' in checkpoint else checkpoint
    G.load_state_dict(state_dict)
    G.eval()
    
    os.makedirs(outdir, exist_ok=True)
    
    seeds_list = [int(s) for s in seeds.split(',')]
    
    for seed in seeds_list:
        print(f"Generating image for seed {seed}...")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate z
        z_dim = G.z_dim
        z = torch.randn(1, z_dim).to(device)
        
        with torch.no_grad():
            # Pass truncation_psi if supported by model's forward
            img = G(z, truncation_psi=truncation)
            
        # Save image
        save_path = os.path.join(outdir, f"seed{seed}.png")
        # Normalize from [-1, 1] to [0, 1] for saving
        save_image(img, save_path, normalize=True, value_range=(-1, 1))
        print(f"Saved to {save_path}")

    print("Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--truncation", type=float, default=0.7, help="Truncation psi")
    parser.add_argument("--outdir", default="work_dirs/inference_output", help="Output directory")
    
    args = parser.parse_args()
    
    run_inference(args.config, args.checkpoint, args.seeds, args.truncation, args.outdir)
