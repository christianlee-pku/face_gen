import onnxruntime as ort
import numpy as np
import os

class ONNXInferenceWrapper:
    """
    Wrapper for running inference on an exported ONNX StyleGAN3 model.
    """
    def __init__(self, model_path, providers=['CPUExecutionProvider']):
        """
        Args:
            model_path (str): Path to the .onnx model file.
            providers (list): List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def generate(self, z):
        """
        Generate images from latent vectors.
        
        Args:
            z (np.ndarray): Latent vectors of shape (batch_size, z_dim).
            
        Returns:
            np.ndarray: Generated images of shape (batch_size, 3, H, W).
        """
        # Ensure z is float32
        z = z.astype(np.float32)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: z})
        return outputs[0]
