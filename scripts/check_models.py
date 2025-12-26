import torch
from mtsf.models.numeric_forecaster import NumericForecaster
from mtsf.models.multimodal_forecaster import MultimodalForecaster

B, L, D = 4, 96, 7
H = 24

x = torch.randn(B, L, D)
img = torch.randn(B, 1, L, L)

num_model = NumericForecaster(input_dim=D, horizon=H)
mm_model = MultimodalForecaster(input_dim=D, horizon=H)

y_num = num_model(x)
y_mm = mm_model(x, img)

print("Numeric output:", y_num.shape)
print("Multimodal output:", y_mm.shape)
