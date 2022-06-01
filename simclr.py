import math

import torch
import torch.nn as nn
from torch import Tensor


class SimCLR(nn.Module):
    def __init__(self, model, latent_dim: int = 64, hidden_dim: int = 1024, t: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.out_dim = self._discover_output_dim()
        self.latent_dim = latent_dim 
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(self.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            L2Norm()
        )
        self.t = t
        self.eps = eps

    def _discover_output_dim(self):
        x = torch.randn(1, 3, 64, 64).to(next(self.model.parameters()).device)
        state = self.model.training
        self.model.eval()
        with torch.no_grad():
            x = self.model(x)
        self.model.train(state)
        return x.shape[1]
        
    def forward(self, x):
        x = self.model(x)
        z = self.projection(x)
        return z

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=1)

def nt_xent_loss(x1, x2, t: float = 0.1, eps: float = 1e-6):
    """
    Normalized, temparature-scaled cross-entropy loss
    Args:
        x1, x2: Tensors of size (batch_size, dim)
        t: Temperature
    """
    x = torch.cat([x1, x2], dim=0)

    cov = x@x.t().contiguous()
    sim = torch.exp(cov / t)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / t)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss
