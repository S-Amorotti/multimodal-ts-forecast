import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Callable, Tuple


class ETTDataset(Dataset):
    """
    Multivariate input -> univariate target (OT)
    Optional image transform applied on target history window
    """

    def __init__(
        self,
        csv_path: str,
        split: str,                      # "train" | "val" | "test"
        lookback: int,
        horizon: int,
        image_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
    ):
        assert split in {"train", "val", "test"}

        df = pd.read_csv(csv_path)
        df = df.drop(columns=["date"])

        values = df.values.astype(np.float32)
        target_idx = df.columns.get_loc("OT")

        T = len(values)
        train_end = int(T * train_frac)
        val_end = int(T * (train_frac + val_frac))

        if split == "train":
            data = values[:train_end]
        elif split == "val":
            data = values[train_end:val_end]
        else:
            data = values[val_end:]

        # Standardization (fit ONLY on train)
        self.scaler = StandardScaler()
        if split == "train":
            self.scaler.fit(data)
        else:
            train_data = values[:train_end]
            self.scaler.fit(train_data)

        data = self.scaler.transform(data)

        self.X = data
        self.target_idx = target_idx
        self.lookback = lookback
        self.horizon = horizon
        self.image_transform = image_transform

        self.indices = []
        for t in range(lookback, len(self.X) - horizon):
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        t = self.indices[idx]

        x_hist = self.X[t - self.lookback: t]  # (L, num_features)
        y_future = self.X[t: t + self.horizon, self.target_idx]  # (H,)

        x_tensor = torch.from_numpy(x_hist)
        y_tensor = torch.from_numpy(y_future)

        img_tensor = None
        if self.image_transform is not None:
            target_hist = x_hist[:, self.target_idx]        # (L,)
            img = self.image_transform(target_hist)
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, L, L)

        return x_tensor, y_tensor, img_tensor
