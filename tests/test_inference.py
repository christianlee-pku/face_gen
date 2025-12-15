import pytest
import numpy as np
import torch
from src.apis.inference import ONNXInferenceWrapper

# Mock ONNX model creation for testing
def create_dummy_onnx_model(path, z_dim=512, img_size=256):
    import torch.nn as nn
    
    class DummyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(z_dim, 3 * img_size * img_size)
        def forward(self, x):
            return self.linear(x).view(-1, 3, img_size, img_size)
    
    model = DummyGenerator()
    dummy_input = torch.randn(1, z_dim)
    torch.onnx.export(
        model, 
        dummy_input, 
        path, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory):
    # Create a temporary ONNX model
    fn = tmp_path_factory.mktemp("models") / "dummy_stylegan.onnx"
    create_dummy_onnx_model(str(fn))
    return str(fn)

def test_inference_wrapper_init(onnx_model_path):
    """Test initialization of the wrapper."""
    wrapper = ONNXInferenceWrapper(onnx_model_path)
    assert wrapper.session is not None
    assert wrapper.input_name == 'input'
    assert wrapper.output_name == 'output'

def test_inference_wrapper_generate(onnx_model_path):
    """Test image generation."""
    wrapper = ONNXInferenceWrapper(onnx_model_path)
    z_dim = 512
    batch_size = 2
    
    z = np.random.randn(batch_size, z_dim).astype(np.float32)
    images = wrapper.generate(z)
    
    assert isinstance(images, np.ndarray)
    assert images.shape == (batch_size, 3, 256, 256)
    assert images.dtype == np.float32

def test_inference_wrapper_invalid_path():
    """Test error handling for missing model."""
    with pytest.raises(FileNotFoundError):
        ONNXInferenceWrapper("non_existent_model.onnx")