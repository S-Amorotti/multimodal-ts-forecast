# src/mtsf/training/run_experiment.py

import os
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from mtsf.training.collate import collate_optional_image
from mtsf.datasets.ett import ETTDataset
from mtsf.transforms.gaf import gaf_gramian_angular_summation_field
from mtsf.transforms.shuffled import ShuffledImageTransform
from mtsf.models.numeric_forecaster import NumericForecaster
from mtsf.models.multimodal_forecaster import MultimodalForecaster
from mtsf.training.engine import train_one_epoch, evaluate


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lookback = cfg["data"]["lookback"]
    horizon = cfg["data"]["horizon"]

    use_images = bool(cfg["model"]["use_images"])
    shuffle_images = bool(cfg["model"].get("shuffle_images", False))

    # Ensure output directory exists
    save_path = cfg["train"]["save_path"]
    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Select transform
    if use_images:
        transform = gaf_gramian_angular_summation_field
        if shuffle_images:
            transform = ShuffledImageTransform(transform)
    else:
        transform = None

    mode = "multimodal" if use_images else "numeric"
    if shuffle_images:
        mode += "_shuffled"
    print(f"Running mode: {mode} | seed={seed} | device={device}")

    # ----------------
    # Datasets
    # ----------------
    train_ds = ETTDataset(
        csv_path=cfg["data"]["path"],
        split="train",
        lookback=lookback,
        horizon=horizon,
        image_transform=transform,
    )

    val_ds = ETTDataset(
        csv_path=cfg["data"]["path"],
        split="val",
        lookback=lookback,
        horizon=horizon,
        image_transform=transform,
    )

    # ----------------
    # DataLoaders
    # ----------------
    batch_size = int(cfg["train"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_optional_image,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_optional_image,
    )

    # ----------------
    # Model
    # ----------------
    input_dim = train_ds.X.shape[1]

    if use_images:
        model = MultimodalForecaster(
            input_dim=input_dim,
            horizon=horizon,
        )
    else:
        model = NumericForecaster(
            input_dim=input_dim,
            horizon=horizon,
        )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
    )

    # ----------------
    # Training
    # ----------------
    best_mae = float("inf")
    epochs = int(cfg["train"]["epochs"])

    for epoch in range(epochs):
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_path)

    print("Best Val MAE:", best_mae)

    # ----------------
    # Test evaluation
    # ----------------
    print("Evaluating on test set...")

    test_ds = ETTDataset(
        csv_path=cfg["data"]["path"],
        split="test",
        lookback=lookback,
        horizon=horizon,
        image_transform=transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_optional_image,
    )

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_mae, test_rmse = evaluate(model, test_loader, device)

    print(f"TEST MAE: {test_mae:.4f} | TEST RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    run(args.config)
