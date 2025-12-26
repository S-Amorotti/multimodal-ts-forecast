import torch
import torch.nn as nn


class GAFImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, out_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, 1, L, L)
        returns: (B, out_dim)
        """
        h = self.net(img).flatten(1)
        return self.fc(h)
