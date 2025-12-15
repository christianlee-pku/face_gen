import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from src.apis.postprocessing import InvisibleWatermarker

def demo_real_image_watermark():
    # Configuration
    input_image_path = "data/celeba/img_align_celeba/000001.jpg"
    output_dir = "work_dirs/demo_real_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading image from {input_image_path}...")
    
    # 1. Load and Preprocess Image
    # Open image
    pil_img = Image.open(input_image_path).convert('RGB')
    
    # Resize to 256x256 as expected by our model/watermarker default
    pil_img = pil_img.resize((256, 256))
    
    # Convert to numpy array and normalize to [-1, 1] for the watermarker
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # HWC -> CHW
    img_tensor = img_tensor * 2.0 - 1.0 # [0, 1] -> [-1, 1]
    
    # 2. Initialize Watermarker
    # We use a slightly higher strength here to ensure the "difference" is visible in the demo
    # Standard might be 0.01-0.05, we use 0.1 for clear visualization of the noise pattern
    strength = 0.5 
    print(f"Initializing InvisibleWatermarker with strength={strength}...")
    watermarker = InvisibleWatermarker(input_shape=(3, 256, 256), signature_strength=strength)
    
    # 3. Apply Watermark
    print("Applying watermark...")
    watermarked_tensor = watermarker.apply(img_tensor)
    
    # 4. Detect Watermark
    print("Detecting watermark...")
    is_detected = watermarker.detect(watermarked_tensor)
    print(f"Watermark Detected? {is_detected[0]}")
    
    # 5. Post-process for Saving/Visualizing
    # Convert back to [0, 1] range and HWC
    # watermarked_tensor usually comes back as (1, C, H, W) because apply() adds batch dim if missing or preserves it
    if watermarked_tensor.dim() == 4:
        watermarked_tensor = watermarked_tensor.squeeze(0)
        
    original_np = ((img_tensor.permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
    watermarked_np = ((watermarked_tensor.permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
    
    # 6. Calculate Difference
    diff = np.abs(watermarked_np - original_np)
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")
    
    # Amplify difference for visualization (Boost 10x)
    diff_boosted = diff * 10.0
    diff_boosted = np.clip(diff_boosted, 0, 1)
    
    # 7. Save Images
    original_out = os.path.join(output_dir, "01_original.png")
    watermarked_out = os.path.join(output_dir, "02_watermarked.png")
    diff_out = os.path.join(output_dir, "03_difference_x10.png")
    
    plt.imsave(original_out, original_np)
    plt.imsave(watermarked_out, watermarked_np)
    plt.imsave(diff_out, diff_boosted)
    
    print(f"\nSaved images to {output_dir}:")
    print(f"1. Original: {original_out}")
    print(f"2. Watermarked: {watermarked_out}")
    print(f"3. Difference (Boosted x10): {diff_out}")
    
    print("\nCheck the difference image to see the watermark noise pattern clearly.")

if __name__ == "__main__":
    demo_real_image_watermark()
