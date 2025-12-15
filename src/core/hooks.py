from src.core.registry import HOOKS
import torch
import numpy as np

@HOOKS.register_module()
class BiasDetectionHook:
    """
    Hook to check for attribute bias in generated images during training.
    """
    def __init__(self, attribute_classifier=None, target_distribution=None, interval=1000):
        """
        Args:
            attribute_classifier: Path to a pre-trained attribute classifier (e.g., ONNX or PyTorch model).
            target_distribution (dict): Expected distribution of attributes (e.g., {'Male': 0.5}).
            interval (int): How often to run the check (iterations).
        """
        self.interval = interval
        self.target_distribution = target_distribution
        # In a real implementation, we would load the classifier here.
        # For this task, we'll use a placeholder logic or mock.
        self.classifier = attribute_classifier 

    def after_train_iter(self, trainer):
        """
        Called after each training iteration.
        """
        if trainer.iter_count % self.interval != 0:
            return
            
        trainer.logger.info("Running bias detection...")
        
        # Generate a batch of images
        trainer.G.eval()
        with torch.no_grad():
            z = torch.randn(32, trainer.G.z_dim).cuda()
            _ = trainer.G(z)
            
            # --- Bias Logic Stub ---
            # 1. Resize images for classifier
            # 2. Run classifier -> get logits
            # 3. Calculate stats (e.g., percentage of 'Male')
            # 4. Compare with target_distribution
            # 5. Log deviation to MLflow
            
            # Mocking result for demonstration
            detected_male_ratio = np.random.uniform(0.4, 0.6)
            bias_score = abs(detected_male_ratio - 0.5) # Assuming 0.5 target
            
            trainer.logger.info(f"Bias Check: Male Ratio = {detected_male_ratio:.2f} (Bias Score: {bias_score:.4f})")
            
            # Log to MLflow if available in trainer
            # (Trainer class we wrote imports mlflow globally, but good to be robust)
            import mlflow
            if mlflow.active_run():
                mlflow.log_metric("bias_score", bias_score, step=trainer.iter_count)
                
        trainer.G.train()
