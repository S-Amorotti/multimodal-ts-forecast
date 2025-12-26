from mtsf.datasets.ett import ETTDataset
from mtsf.transforms.gaf import gaf_gramian_angular_summation_field

ds = ETTDataset(
    csv_path="data/ETTm1.csv",
    split="train",
    lookback=96,
    horizon=24,
    image_transform=gaf_gramian_angular_summation_field,
)

x, y, img = ds[0]

print("x:", x.shape)       # (96, 7)
print("y:", y.shape)       # (24,)
print("img:", img.shape)   # (1, 96, 96)
