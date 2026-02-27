import argparse
import math
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

# Config 
RGB_BANDS   = (3, 2, 1)      # band indices (1-based): R=B3, G=B2, B=B1
TILES_ROOT  = Path("data/tiles_tif")
MASKS_ROOT  = Path("data/masks_tif")
OUT_DIR     = Path("data/grid_previews")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="both", choices=["train", "test", "both"])
    p.add_argument("--cols",   type=int, default=0,   help="Cols per row (0=auto)")
    p.add_argument("--thumb",  type=int, default=112, help="Thumbnail px per tile cell")
    return p.parse_args()

def norm_percentile(arr: np.ndarray, lo=2, hi=98) -> np.ndarray:
    """Stretch a 2-D float array to 0–255 uint8."""
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo_v = np.percentile(valid, lo)
    hi_v = np.percentile(valid, hi)
    stretched = np.clip((arr - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)
    return (stretched * 255).astype(np.uint8)

def read_rgb(tif_path: Path, thumb: int) -> np.ndarray:
    """Read a GeoTIFF and return a (thumb, thumb, 3) uint8 RGB thumbnail."""
    with rasterio.open(tif_path) as src:
        n = src.count
        # Pick available bands (fall back to band 1 if fewer than 3)
        def read_band(idx):
            b = min(idx, n)
            return src.read(b, out_shape=(thumb, thumb),
                            resampling=rasterio.enums.Resampling.average
                            ).astype(np.float32)

        r = norm_percentile(read_band(RGB_BANDS[0]))
        g = norm_percentile(read_band(RGB_BANDS[1]))
        b = norm_percentile(read_band(RGB_BANDS[2]))

    return np.stack([r, g, b], axis=-1)

def read_mask(tif_path: Path, thumb: int) -> np.ndarray:
    """Read a 1-band mask GeoTIFF and return a (thumb, thumb) bool array."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1, out_shape=(thumb, thumb),
                       resampling=rasterio.enums.Resampling.nearest)
    return arr > 127

def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend mask dots onto RGB:
      - tree dots  → bright green
      - background → dimmed tile
    """
    overlay = (rgb * 0.55).astype(np.uint8)
    overlay[mask, 0] = 30
    overlay[mask, 1] = 220
    overlay[mask, 2] = 60
    return overlay

def make_gap(thumb: int, w: int = 2) -> np.ndarray:
    """Thin white vertical separator."""
    return np.full((thumb, w, 3), 240, dtype=np.uint8)

def render_grid(split: str, thumb: int, cols: int):
    tile_dir = TILES_ROOT / split
    mask_dir = MASKS_ROOT / split

    tiles = sorted(tile_dir.glob("tile_*.tif"))
    if not tiles:
        print(f"  No tiles found in {tile_dir}")
        return

    n = len(tiles)
    if cols == 0:
        cols = max(6, math.ceil(math.sqrt(n * 2)))   # 2 panels per tile
    # Each *tile pair* takes 2 cell-widths + gap; arrange tile-pairs in rows
    pair_cols = max(1, cols // 2)
    rows = math.ceil(n / pair_cols)

    cell_w = thumb * 2 + 2   # [tile | gap | overlay]
    cell_h = thumb + 20       # extra for row label

    canvas_w = pair_cols * (cell_w + 4)
    canvas_h = rows * cell_h

    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)  # dark bg

    print(f"\n  [{split.upper()}]  {n} tiles  →  {rows} rows × {pair_cols} pairs/row")

    for i, tile_path in enumerate(tiles):
        row_i = i // pair_cols
        col_i = i %  pair_cols

        rgb = read_rgb(tile_path, thumb)

        mask_path = mask_dir / tile_path.name
        if mask_path.exists():
            mask = read_mask(mask_path, thumb)
            overlay = make_overlay(rgb, mask)
        else:
            overlay = np.full_like(rgb, 80)   # grey placeholder

        gap = make_gap(thumb)
        pair = np.concatenate([rgb, gap, overlay], axis=1)  # (thumb, cell_w, 3)

        y0 = row_i * cell_h
        x0 = col_i * (cell_w + 4)
        canvas[y0 : y0 + thumb, x0 : x0 + cell_w] = pair

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n} pairs rendered ...", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"grid_{split}.png"

    dpi = 150
    fig_w = canvas_w / dpi
    fig_h = (canvas_h + 60) / dpi     # + space for title

    fig, ax = plt.subplots(figsize=(max(fig_w, 8), max(fig_h, 4)), dpi=dpi)
    ax.imshow(canvas)
    ax.axis("off")

    title = (f"{split.upper()} — {n} tiles   "
             f"│  Left: RGB (B3·B2·B1)   │  Right: mask overlay (green dots = trees)")
    ax.set_title(title, fontsize=max(7, int(fig_w * 0.8)),
                 fontweight="bold", color="white",
                 pad=6, loc="left")
    fig.patch.set_facecolor("#1e1e1e")

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out_path}  ({canvas_w}×{canvas_h} px)")
