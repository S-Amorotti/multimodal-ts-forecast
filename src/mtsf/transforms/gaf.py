import numpy as np


def gaf_gramian_angular_summation_field(
    x: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Gramian Angular Summation Field (GASF)

    Parameters
    ----------
    x : np.ndarray
        Shape (L,) univariate time series window
    eps : float
        Numerical stability constant

    Returns
    -------
    np.ndarray
        Shape (L, L) GASF image
    """
    x = x.astype(np.float32)

    # Normalize to [-1, 1]
    min_x = x.min()
    max_x = x.max()
    x_norm = 2.0 * (x - min_x) / (max_x - min_x + eps) - 1.0
    x_norm = np.clip(x_norm, -1.0, 1.0)

    # Polar encoding
    phi = np.arccos(x_norm)  # in [0, pi]

    # GASF
    gaf = np.cos(phi[:, None] + phi[None, :])

    return gaf.astype(np.float32)
