import torch
import torch.nn.functional as F
from src.core.registry import MODELS

@MODELS.register_module()
class LogisticLoss(torch.nn.Module):
    def forward(self, prediction, target_is_real):
        if target_is_real:
            return F.softplus(-prediction).mean()
        else:
            return F.softplus(prediction).mean()

@MODELS.register_module()
class R1Penalty(torch.nn.Module):
    def __init__(self, gamma=10.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, real_pred, real_img):
        # We need gradients w.r.t input
        if not real_img.requires_grad:
             real_img.requires_grad = True
             
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return (self.gamma / 2) * grad_penalty