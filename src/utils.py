import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter

def norm_band(arr: np.ndarray) -> np.ndarray:
    """Percentile-based normalization of a single raster band to [0, 1]."""
    arr = arr.astype(np.float32)
    valid = arr > 0
    lo, hi = (np.percentile(arr[valid], (2, 98)) if np.any(valid) else (0.0, 1.0))
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

def read_tif(path: str) -> np.ndarray:
    """Read up to 3 bands from a GeoTIFF -> normalised float32 (C, H, W)."""
    with rasterio.open(path) as src:
        bands = [norm_band(src.read(i)) for i in range(1, min(4, src.count + 1))]
    return np.stack(bands, axis=0)

def tensor_to_rgb(img_tensor) -> np.ndarray:
    """
    (C, H, W) tensor with bands [B, G, R] -> (H, W, 3) float32 [R, G, B].
    Safe to pass either a torch.Tensor or a numpy array.
    """
    img = img_tensor.cpu().numpy() if torch.is_tensor(img_tensor) else img_tensor
    return np.clip(np.stack([img[2], img[1], img[0]], axis=-1), 0, 1)

def point_mask_to_density(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Binary annotation mask -> Gaussian density map (integral = tree count)."""
    density = np.zeros(mask.shape, dtype=np.float32)
    density[mask > 0] = 1.0
    return gaussian_filter(density, sigma=sigma)

def density_to_binary(density: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Threshold a density map to a binary presence/absence mask."""
    return (density > threshold).astype(np.uint8)

def count_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Absolute error between integrated predicted and GT tree count."""
    return float(abs(pred.sum() - gt.sum()))

def count_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Root-squared error between integrated predicted and GT tree count."""
    return float(np.sqrt((pred.sum() - gt.sum()) ** 2))
