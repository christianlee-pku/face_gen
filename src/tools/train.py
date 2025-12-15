import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.core.registry import DATASETS, MODELS, PIPELINES
from src.core.config import Config
from src.core.logging import get_logger
import mlflow
import numpy as np # Added for evaluate()

# Register modules

class Trainer:
    """
    Trainer class for StyleGAN3 model.
    Encapsulates model building, data loading, training loop, validation, resume, and fine-tuning.
    Device-agnostic (supports CUDA, MPS, CPU).
    """
    def __init__(self, cfg_path):
        self.cfg = Config.fromfile(cfg_path)
        self.logger = get_logger("face_gen_train")
        self.work_dir = self.cfg.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Detect Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._init_tracking()
        self._init_data()
        self._init_model()
        self._init_optimization()
        
        # State
        self.start_epoch = 0
        
        # Resume / Fine-tune logic
        resume_from_cfg = self.cfg._cfg_dict.get('resume_from')
        if resume_from_cfg:
            self.resume_from_checkpoint(resume_from_cfg)
        
        load_from_cfg = self.cfg._cfg_dict.get('load_from')
        if load_from_cfg:
            self.load_weights(load_from_cfg)
        
    def _init_tracking(self):
        # Resolve tracking URI relative to CWD if strictly local to avoid root path issues
        tracking_uri = self.cfg.mlflow.tracking_uri
        if not tracking_uri.startswith("file:") and not tracking_uri.startswith("http"):
             tracking_uri = "file://" + os.path.abspath(tracking_uri)
             
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        
    def _init_data(self):
        self.logger.info("Initializing data...")
        if isinstance(self.cfg.train_pipeline, list):
            from src.datasets.pipelines import Compose
            pipeline = Compose(self.cfg.train_pipeline)
        else:
            pipeline = PIPELINES.build(self.cfg.train_pipeline)
            
        dataset = DATASETS.build(self.cfg.dataset, pipeline=pipeline)
        
        # Disable pin_memory for MPS to avoid warnings
        use_pin_memory = True
        if self.device.type == 'mps':
            use_pin_memory = False
            
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.num_workers,
            pin_memory=use_pin_memory
        )
        
    def _init_model(self):
        self.logger.info(f"Initializing models on {self.device}...")
        self.G = MODELS.build(self.cfg.model.generator).to(self.device)
        self.D = MODELS.build(self.cfg.model.discriminator).to(self.device)
        
    def _init_optimization(self):
        self.logger.info("Initializing optimization...")
        self.g_optim = optim.Adam(self.G.parameters(), lr=self.cfg.lr, betas=(0.0, 0.99))
        self.d_optim = optim.Adam(self.D.parameters(), lr=self.cfg.lr, betas=(0.0, 0.99))
        
        self.loss_logistic = MODELS.build(dict(type='LogisticLoss'))
        self.loss_r1 = MODELS.build(dict(type='R1Penalty'))
        
    def resume_from_checkpoint(self, checkpoint_path):
        """Resume full training state."""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.g_optim.load_state_dict(checkpoint['g_optim'])
        self.d_optim.load_state_dict(checkpoint['d_optim'])
        self.start_epoch = checkpoint['epoch'] + 1 # Resume from next epoch

    def load_weights(self, checkpoint_path):
        """Load model weights only (fine-tuning)."""
        self.logger.info(f"Loading weights for fine-tuning from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        
    def evaluate(self, epoch):
        """
        Evaluation loop to calculate metrics like FID/KID.
        (Placeholder for actual FID/KID calculation)
        """
        self.logger.info(f"Running evaluation for epoch {epoch}...")
        self.G.eval()
        
        # In a real scenario, this would involve:
        # 1. Generating a large number of fake images.
        # 2. Loading a real image dataset (e.g., from validation split).
        # 3. Computing feature representations using a pre-trained Inception network.
        # 4. Calculating FID/KID scores.

        with torch.no_grad():
            # Generate dummy fake images
            num_samples_for_eval = 32 # Can be configured
            z = torch.randn(num_samples_for_eval, self.G.z_dim).to(self.device)
            fake_imgs = self.G(z)
            
            # Simulate FID/KID scores
            # These are random values for demonstration purposes
            fid_score = np.random.uniform(10.0, 50.0)
            kid_score = np.random.uniform(0.01, 0.05)
            
            self.logger.info(f"Epoch {epoch}: Simulated FID = {fid_score:.2f}, KID = {kid_score:.4f}")
            
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics({
                    'eval_fid': fid_score,
                    'eval_kid': kid_score
                }, step=epoch)
                
            # Log artifacts: save some generated images
            from torchvision.utils import save_image
            output_dir = os.path.join(self.work_dir, "eval_images")
            os.makedirs(output_dir, exist_ok=True)
            save_image(fake_imgs, os.path.join(output_dir, f"generated_epoch_{epoch}.png"), normalize=True, nrow=8)

        self.G.train()

    def train(self):
        self.logger.info(f"Starting training for {self.cfg.total_epochs} epochs...")
        
        # Log all config parameters to MLflow
        with mlflow.start_run():
            mlflow.log_params(self.cfg._cfg_dict)
            
            for epoch in range(self.start_epoch, self.cfg.total_epochs):
                for i, real_img in enumerate(self.dataloader):
                    start_time = time.time()
                    
                    real_img = real_img.to(self.device)
                    # Enable gradients for R1 penalty
                    real_img.requires_grad = True
                    batch_size = real_img.size(0)
                    
                    data_time = time.time() - start_time
                    
                    # --- Train Discriminator ---
                    self.D.zero_grad()
                    z = torch.randn(batch_size, self.G.z_dim).to(self.device)
                    fake_img = self.G(z).detach()
                    
                    real_pred = self.D(real_img)
                    fake_pred = self.D(fake_img)
                    
                    d_loss_real = self.loss_logistic(real_pred, True)
                    d_loss_fake = self.loss_logistic(fake_pred, False)
                    d_loss_main = d_loss_real + d_loss_fake
                    
                    # R1 Penalty
                    r1_loss = self.loss_r1(real_pred, real_img)
                    d_loss = d_loss_main + r1_loss
                    
                    d_loss.backward()
                    
                    # Calculate Grad Norm
                    d_grad_norm = 0.0
                    for p in self.D.parameters():
                        if p.grad is not None:
                            d_grad_norm += p.grad.data.norm(2).item() ** 2
                    d_grad_norm = d_grad_norm ** 0.5
                    
                    self.d_optim.step()
                    
                    # --- Train Generator ---
                    self.G.zero_grad()
                    z = torch.randn(batch_size, self.G.z_dim).to(self.device)
                    fake_img = self.G(z)
                    fake_pred = self.D(fake_img)
                    
                    g_loss = self.loss_logistic(fake_pred, True)
                    g_loss.backward()
                    
                    # Calculate Grad Norm
                    g_grad_norm = 0.0
                    for p in self.G.parameters():
                        if p.grad is not None:
                            g_grad_norm += p.grad.data.norm(2).item() ** 2
                    g_grad_norm = g_grad_norm ** 0.5
                    
                    self.g_optim.step()
                    
                    batch_time = time.time() - start_time
                    
                    # --- Logging ---
                    if i % self.cfg.log_interval == 0:
                        # Memory usage
                        mem_usage = 0
                        if self.device.type == 'cuda':
                            mem_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
                        elif self.device.type == 'mps':
                                                try:
                                                    mem_usage = torch.mps.current_allocated_memory() / 1024 / 1024
                                                except Exception:
                                                    pass                        
                        log_msg = (
                            f"mode: train, epoch: {epoch+1}, iter: {i}/{len(self.dataloader)}, lr: {self.cfg.lr:.5f}, "
                            f"memory: {int(mem_usage)}MB, data_time: {data_time:.5f}s, "
                            f"loss_g: {g_loss.item():.5f}, loss_d: {d_loss.item():.5f}, "
                            f"loss_d_real: {d_loss_real.item():.5f}, loss_d_fake: {d_loss_fake.item():.5f}, "
                            f"loss_r1: {r1_loss.item():.5f}, "
                            f"grad_norm_g: {g_grad_norm:.5f}, grad_norm_d: {d_grad_norm:.5f}, "
                            f"time: {batch_time:.5f}s"
                        )
                        self.logger.info(log_msg)
                        
                        # Use global_step as total iters for continuity
                        global_step = epoch * len(self.dataloader) + i
                        mlflow.log_metrics({
                            'loss_g': g_loss.item(),
                            'loss_d': d_loss.item(),
                            'loss_d_real': d_loss_real.item(),
                            'loss_d_fake': d_loss_fake.item(),
                            'loss_r1': r1_loss.item(),
                            'grad_norm_g': g_grad_norm,
                            'grad_norm_d': d_grad_norm,
                            'data_time': data_time,
                            'batch_time': batch_time,
                            'epoch': epoch
                        }, step=global_step)
                
                # --- Checkpointing (End of Epoch) ---
                if (epoch + 1) % self.cfg.checkpoint_interval == 0:
                    self._save_checkpoint(epoch)
                    
                # --- Evaluation ---
                if (epoch + 1) % self.cfg.eval_interval == 0:
                    self.evaluate(epoch)
    
    def _save_checkpoint(self, epoch):
        save_path = os.path.join(self.work_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'g_optim': self.g_optim.state_dict(),
            'd_optim': self.d_optim.state_dict(),
            'epoch': epoch
        }, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()
