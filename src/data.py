import os
import glob
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from .utils import read_tif, point_mask_to_density

class TreeDataset(Dataset):
    """Paired image + density-map dataset for tree orchard detection."""

    def __init__(self, img_dir: str, mask_dir: str, gaussian_sigma: float = 2.0):
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir,  '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
        self.gaussian_sigma = gaussian_sigma
        assert len(self.img_paths) > 0, f'No .tif files found in {img_dir}'
        assert len(self.img_paths) == len(self.mask_paths), (
            f'Image/mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}'
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = read_tif(self.img_paths[idx])                         # (C,H,W)
        with rasterio.open(self.mask_paths[idx]) as src:
            raw_mask = src.read(1)
        binary_mask = (raw_mask > 0).astype(np.uint8)              # (H,W)
        density     = point_mask_to_density(binary_mask, sigma=self.gaussian_sigma)  # (H,W)
        return (
            torch.tensor(img,         dtype=torch.float32),         # (C,H,W)
            torch.tensor(density,     dtype=torch.float32).unsqueeze(0),  # (1,H,W)
            torch.tensor(binary_mask, dtype=torch.uint8),           # (H,W) for viz
        )
