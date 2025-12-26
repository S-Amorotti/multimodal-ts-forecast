import torch
import torch.nn as nn
from .numeric_encoder import NumericTransformerEncoder
from .forecast_heads import ForecastHead


class NumericForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 64,
    ):
        super().__init__()

        self.encoder = NumericTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
        )

        self.head = ForecastHead(
            in_dim=d_model,
            horizon=horizon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)
