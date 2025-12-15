from src.apis.inference import ONNXInferenceWrapper
import argparse
import numpy as np
from PIL import Image
import os

def run_inference(model_path, output_path):
    """
    Run inference using the exported ONNX model.
    """
    print(f"Loading model from {model_path}...")
    wrapper = ONNXInferenceWrapper(model_path)
    
    # Generate random latent vector
    # StyleGAN3 input is (1, 512)
    z = np.random.randn(1, 512).astype(np.float32)
    
    print("Generating image...")
    output = wrapper.generate(z)
    
    # Output shape is (1, 3, H, W), typically float32 [-1, 1]
    # Post-process to uint8 [0, 255]
    img_tensor = output[0] # (3, H, W)
    
    # Normalize from [-1, 1] to [0, 1]
    img_tensor = (img_tensor + 1) / 2
    img_tensor = np.clip(img_tensor, 0, 1)
    
    # Transpose to (H, W, 3)
    img_array = (img_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Save image
    print(f"Saving output to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(img_array).save(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .onnx model file")
    parser.add_argument("--out", help="Output image path", default="work_dirs/generated_face.png")
    args = parser.parse_args()
    
    run_inference(args.model, args.out)
