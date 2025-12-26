import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    def __init__(self, in_dim: int, horizon: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
