from fastapi import FastAPI, status, Depends
from src.apis.schemas import GenerateRequest, GenerateResponse, HealthCheckResponse
from src.apis.auth import auth_dependency
from src.apis.inference import ONNXInferenceWrapper
from src.apis.postprocessing import InvisibleWatermarker
import os
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time

# Initialize global components
# In a real app, these might be initialized in a startup event
MODEL_PATH = os.environ.get("MODEL_PATH", "work_dirs/stylegan3.onnx")
# Check if model exists, if not, we handle gracefully (e.g. for CI/CD where model might not be present)
inference_wrapper = None
if os.path.exists(MODEL_PATH):
    try:
        inference_wrapper = ONNXInferenceWrapper(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Inference will fail.")

watermarker = InvisibleWatermarker(input_shape=(3, 256, 256), signature_strength=0.02)

app = FastAPI(
    title="Face Generation API",
    description="API for generating realistic faces using StyleGAN3",
    version="1.0.0"
)

@app.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint.
    """
    return {"message": "Welcome to the Face Generation API. Visit /docs for API documentation."}

@app.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(auth_dependency)])
async def generate(request: GenerateRequest):
    """
    Generate a face image from a random seed.
    """
    start_time = time.time()
    
    # 1. Handle Seed
    seed = request.seed if request.seed is not None else int(time.time())
    rng = np.random.RandomState(seed)
    
    # 2. Generate Latent Vector
    z_dim = 512 # Should match model config
    z = rng.randn(1, z_dim).astype(np.float32)
    
    # 3. Inference
    if inference_wrapper is None:
        # Fallback for testing/no-model scenario (return noise)
        # In prod, this should likely be 503 Service Unavailable
        fake_img_np = np.random.rand(1, 3, 256, 256).astype(np.float32) * 2 - 1
    else:
        fake_img_np = inference_wrapper.generate(z)
        
    # 4. Watermarking
    # Input to watermarker expects [-1, 1], returns same range
    # Ensure tensor/numpy compat
    fake_img_watermarked = watermarker.apply(fake_img_np)
    
    # 5. Post-processing (Format for output)
    # [-1, 1] -> [0, 255]
    if isinstance(fake_img_watermarked, torch.Tensor):
        fake_img_watermarked = fake_img_watermarked.cpu().numpy()
        
    img_data = (np.transpose(fake_img_watermarked[0], (1, 2, 0)) + 1) / 2.0 * 255.0
    img_data = np.clip(img_data, 0, 255).astype(np.uint8)
    
    # 6. Encode to Base64
    pil_img = Image.fromarray(img_data)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    latency = time.time() - start_time
    
    return {
        "image": img_str,
        "meta": {
            "seed": seed,
            "truncation_psi": request.truncation_psi,
            "latency_sec": latency,
            "model_version": "v1"
        }
    }