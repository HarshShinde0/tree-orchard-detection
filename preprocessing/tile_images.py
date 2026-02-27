import sys, io, gc, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Tile satellite images into 512×512 patches with overlap.
Each tile is saved as PNG using band combination R=B3, G=B2, B=B1.
Images are split into train/ and test/ based on the SPLIT_MAP config.
Tiles that are mostly empty (nodata / black) are skipped.
A visual grid preview is also generated.
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import gc
import matplotlib.pyplot as plt
from pathlib import Path

# Config

TILE_SIZE = 512            # pixels
OVERLAP   = 128            # pixels of overlap between tiles (handles border trees)
MIN_VALID_FRAC = 0.3       # skip tiles with less than 30% valid pixels

# RGB channel mapping:  R=Band3, G=Band2, B=Band1
# Band order in .tif:   B1=Blue, B2=NIR, B3=Green, B4=Red
RGB_BANDS = (3, 2, 1)      # (R_channel, G_channel, B_channel) using band indices

# Train / Test split mapping
# Key = filename stem (without .tif), Value = "train" or "test"
SPLIT_MAP = {
    "tree_orchard_count_new0": "train",   # 153 tiles → for annotation & training
    "tree_orchard_count_new1": "test",    # 40 tiles  → for inference only
}

# Input / output
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TIF_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
OUT_ROOT = os.path.join(DATA_DIR, "data", "tiles")

def norm_percentile(arr, lo_pct=2, hi_pct=98):
    """Percentile-stretch a 2D array to 0–255 uint8."""
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = np.percentile(valid, lo_pct)
    hi = np.percentile(valid, hi_pct)
    stretched = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    return (stretched * 255).astype(np.uint8)

def tile_one_image(tif_path):
    """
    Tile a single GeoTIFF into TILE_SIZE × TILE_SIZE patches.
    
    Tiles are saved into train/ or test/ based on SPLIT_MAP:
      data/tiles/train/tile_<name>_R{row}_C{col}.png
      data/tiles/test/tile_<name>_R{row}_C{col}.png
    
    Returns:
      tile_info  – list of dicts with tile metadata
    """
    basename = Path(tif_path).stem
    split = SPLIT_MAP.get(basename, "train")  # default to train if not specified
    
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    tile_info = []
    stride = TILE_SIZE - OVERLAP

    t0 = time.time()

    with rasterio.open(tif_path) as src:
        img_h, img_w = src.height, src.width
        n_bands = src.count

        print(f"\n  Image size     : {img_h} × {img_w} ({n_bands} bands)")
        print(f"  Split          : {split.upper()}")
        print(f"  Tile size      : {TILE_SIZE} × {TILE_SIZE}")
        print(f"  Overlap        : {OVERLAP} px")
        print(f"  Stride         : {stride} px")

        # Calculate grid dimensions
        n_rows = max(1, (img_h - OVERLAP) // stride)
        n_cols = max(1, (img_w - OVERLAP) // stride)

        # Ensure we cover the full image (add extra row/col if needed)
        if (n_rows - 1) * stride + TILE_SIZE < img_h:
            n_rows += 1
        if (n_cols - 1) * stride + TILE_SIZE < img_w:
            n_cols += 1

        total_tiles = n_rows * n_cols
        print(f"  Grid           : {n_rows} rows × {n_cols} cols = {total_tiles} potential tiles")

        saved = 0
        skipped = 0

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Calculate window position
                y_off = row_idx * stride
                x_off = col_idx * stride

                # Clamp to image bounds
                y_off = min(y_off, max(0, img_h - TILE_SIZE))
                x_off = min(x_off, max(0, img_w - TILE_SIZE))

                # Read window (all bands)
                window = Window(x_off, y_off, TILE_SIZE, TILE_SIZE)
                tile_data = src.read(window=window).astype(np.float32)
                # tile_data shape: (bands, H, W)

                actual_h, actual_w = tile_data.shape[1], tile_data.shape[2]

                # Pad if tile is smaller than TILE_SIZE (edge tiles)
                if actual_h < TILE_SIZE or actual_w < TILE_SIZE:
                    padded = np.zeros((n_bands, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    padded[:, :actual_h, :actual_w] = tile_data
                    tile_data = padded

                # --- Check validity (skip mostly-empty tiles) ---
                pixel_sum = tile_data.sum(axis=0)
                valid_frac = (pixel_sum > 0).sum() / (TILE_SIZE * TILE_SIZE)

                if valid_frac < MIN_VALID_FRAC:
                    skipped += 1
                    continue

                # Save RGB PNG (R=B3, G=B2, B=B1)
                r_ch = norm_percentile(tile_data[RGB_BANDS[0] - 1])
                g_ch = norm_percentile(tile_data[RGB_BANDS[1] - 1])
                b_ch = norm_percentile(tile_data[RGB_BANDS[2] - 1])
                rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)  # (H, W, 3) uint8

                # Filename includes source image name for traceability
                short_name = basename.replace("tree_orchard_count_", "")
                png_name = f"tile_{short_name}_R{row_idx:03d}_C{col_idx:03d}.png"
                png_path = os.path.join(out_dir, png_name)
                plt.imsave(png_path, rgb)

                # Free memory
                del tile_data, r_ch, g_ch, b_ch, rgb

                tile_info.append({
                    "file": basename,
                    "split": split,
                    "row": row_idx,
                    "col": col_idx,
                    "x_off": x_off,
                    "y_off": y_off,
                    "valid_frac": float(valid_frac),
                    "png": png_path,
                })

                saved += 1

                if saved % 25 == 0 or saved == 1:
                    elapsed = time.time() - t0
                    print(f"    Saved {saved} tiles ... (skipped {skipped} empty) [{elapsed:.0f}s]", flush=True)

                # Periodic garbage collection
                if saved % 100 == 0:
                    gc.collect()

        elapsed = time.time() - t0
        print(f"\n  Done: {saved} tiles saved, {skipped} empty tiles skipped [{elapsed:.0f}s]")
        print(f"     RGB PNGs (R=B{RGB_BANDS[0]}, G=B{RGB_BANDS[1]}, B=B{RGB_BANDS[2]}) -> {out_dir}")

    gc.collect()
    return tile_info


def create_grid_preview(tile_infos, tif_path):
    """Create a visual preview showing the tile grid on the full image."""
    basename = Path(tif_path).stem
    split = SPLIT_MAP.get(basename, "train")

    with rasterio.open(tif_path) as src:
        # Read a downscaled version for the preview
        scale = 8
        h, w = src.height // scale, src.width // scale
        r_ch = src.read(RGB_BANDS[0], out_shape=(h, w)).astype(np.float32)
        g_ch = src.read(RGB_BANDS[1], out_shape=(h, w)).astype(np.float32)
        b_ch = src.read(RGB_BANDS[2], out_shape=(h, w)).astype(np.float32)

        rgb = np.stack([
            norm_percentile(r_ch),
            norm_percentile(g_ch),
            norm_percentile(b_ch)
        ], axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.imshow(rgb)

    # Draw tile grid
    for t in tile_infos:
        if t["file"] != basename:
            continue
        x = t["x_off"] / scale
        y = t["y_off"] / scale
        size = TILE_SIZE / scale
        rect = plt.Rectangle((x, y), size, size,
                              linewidth=0.5, edgecolor='cyan',
                              facecolor='none', alpha=0.6)
        ax.add_patch(rect)

    n_tiles = len([t for t in tile_infos if t['file'] == basename])
    ax.set_title(f"Tile Grid — {basename} [{split.upper()}]\n"
                 f"{n_tiles} tiles "
                 f"({TILE_SIZE}×{TILE_SIZE}, overlap={OVERLAP})",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    preview_path = os.path.join(OUT_ROOT, split, f"_grid_preview_{basename}.png")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grid preview -> {preview_path}")


def show_sample_tiles(tile_infos, n=12):
    """Show a sample of tiles in a grid for quick visual check."""
    if not tile_infos:
        return

    sample = tile_infos[:min(n, len(tile_infos))]
    cols = min(4, len(sample))
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, t in enumerate(sample):
        r, c = idx // cols, idx % cols
        img = plt.imread(t["png"])
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"R{t['row']}_C{t['col']}\n{t['valid_frac']:.0%} valid",
                             fontsize=9)
        axes[r, c].axis("off")

    # Hide unused axes
    for idx in range(len(sample), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    basename = tile_infos[0]["file"]
    split = tile_infos[0]["split"]
    preview_path = os.path.join(OUT_ROOT, split, f"_sample_tiles_{basename}.png")
    plt.suptitle(f"Sample Tiles — {basename} [{split.upper()}]", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sample tiles -> {preview_path}")
