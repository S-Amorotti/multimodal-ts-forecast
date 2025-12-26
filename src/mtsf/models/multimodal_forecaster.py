import torch
import torch.nn as nn
from .numeric_encoder import NumericTransformerEncoder
from .image_encoder import GAFImageEncoder
from .forecast_heads import ForecastHead


class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 64,
        d_img: int = 128,
    ):
        super().__init__()

        self.numeric_encoder = NumericTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
        )

        self.image_encoder = GAFImageEncoder(
            out_dim=d_img,
        )

        self.head = ForecastHead(
            in_dim=d_model + d_img,
            horizon=horizon,
        )

    def forward(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        z_num = self.numeric_encoder(x)
        z_img = self.image_encoder(img)
        z = torch.cat([z_num, z_img], dim=-1)
        return self.head(z)
