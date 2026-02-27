import argparse
import gc
import glob
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

#  CONFIG

TILE_SIZE     = 512
OVERLAP       = 128
MIN_VALID_FRAC = 0.30    # skip tiles with <30 % valid (non-zero) pixels
DOT_RADIUS    = 3        # px radius per tree in mask

SPLIT_MAP = {
    "tree_orchard_count_new0": "train",
    "tree_orchard_count_new1": "test",
}

DATA_DIR     = os.path.dirname(os.path.abspath(__file__))
TIF_GLOB     = os.path.join(DATA_DIR, "*.tif")

OUT_TILES    = os.path.join(DATA_DIR, "data", "tiles_tif")
OUT_MASKS    = os.path.join(DATA_DIR, "data", "masks_tif")

# Roboflow XML roots (one per split)
ANNOT_ROOTS  = {
    "train": os.path.join(DATA_DIR, "data", "image", "train"),
    "test":  os.path.join(DATA_DIR, "data", "image", "test"),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-size",   type=int,   default=TILE_SIZE)
    p.add_argument("--overlap",     type=int,   default=OVERLAP)
    p.add_argument("--dot-radius",  type=int,   default=DOT_RADIUS)
    p.add_argument("--skip-tiles",  action="store_true", help="Skip tiling step")
    p.add_argument("--skip-masks",  action="store_true", help="Skip mask step")
    return p.parse_args()

# Tile original GeoTIFFs → GeoTIFF tiles
def tile_one_image(tif_path: str, tile_size: int, overlap: int) -> list[dict]:
    """
    Tile a single GeoTIFF into (tile_size × tile_size) GeoTIFF patches.
    Each output tile carries the correct CRS and affine transform.
    Returns list of metadata dicts for tiles that were saved.
    """
    basename = Path(tif_path).stem
    split    = SPLIT_MAP.get(basename, "train")
    short    = basename.replace("tree_orchard_count_", "")
    out_dir  = Path(OUT_TILES) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stride   = tile_size - overlap
    saved_meta = []
    t0 = time.time()

    with rasterio.open(tif_path) as src:
        img_h, img_w = src.height, src.width
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=tile_size,
            width=tile_size,
            count=src.count,
            dtype=src.dtypes[0],
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        n_rows = max(1, (img_h - overlap) // stride)
        n_cols = max(1, (img_w - overlap) // stride)
        if (n_rows - 1) * stride + tile_size < img_h:
            n_rows += 1
        if (n_cols - 1) * stride + tile_size < img_w:
            n_cols += 1

        print(f"  {basename}  [{split.upper()}]  "
              f"{img_h}×{img_w}  →  {n_rows}×{n_cols} tiles")

        saved = skipped = 0

        for ri in range(n_rows):
            for ci in range(n_cols):
                y_off = min(ri * stride, max(0, img_h - tile_size))
                x_off = min(ci * stride, max(0, img_w - tile_size))

                win  = Window(x_off, y_off, tile_size, tile_size)
                data = src.read(window=win)          # (bands, H, W)

                # Pad edge tiles
                bnd, actual_h, actual_w = data.shape
                if actual_h < tile_size or actual_w < tile_size:
                    padded = np.zeros((bnd, tile_size, tile_size), dtype=data.dtype)
                    padded[:, :actual_h, :actual_w] = data
                    data = padded

                # Skip mostly-empty tiles
                valid_frac = (data.sum(axis=0) > 0).sum() / (tile_size * tile_size)
                if valid_frac < MIN_VALID_FRAC:
                    skipped += 1
                    continue

                # Per-tile affine transform
                tile_transform = src.window_transform(win)

                tile_name = f"tile_{short}_R{ri:03d}_C{ci:03d}.tif"
                tile_path = out_dir / tile_name

                tile_profile = profile.copy()
                tile_profile["transform"] = tile_transform

                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(data)
                    if src.crs:
                        dst.crs = src.crs

                saved_meta.append({
                    "basename": basename,
                    "split":    split,
                    "row": ri, "col": ci,
                    "x_off": x_off, "y_off": y_off,
                    "stem": tile_path.stem,
                    "path": str(tile_path),
                })
                saved += 1

                if saved % 25 == 0 or saved == 1:
                    print(f"    {saved} tiles saved, {skipped} skipped "
                          f"[{time.time()-t0:.0f}s]", flush=True)

        gc.collect()
        print(f"  ✓  {saved} GeoTIFF tiles → {out_dir}  "
              f"({skipped} empty skipped, {time.time()-t0:.0f}s)\n")

    return saved_meta

# Roboflow appends  _png.rf.<hash>  to the original tile stem. Strip it to recover the original stem: tile_new0_R000_C000
_RF_SUFFIX = re.compile(r"_png\.rf\.[0-9a-f]+$", re.IGNORECASE)

def rf_stem_to_tile_stem(rf_stem: str) -> str:
    return _RF_SUFFIX.sub("", rf_stem)


def parse_voc_boxes(xml_path: str) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append({
            "xmin": int(float(b.find("xmin").text)),
            "ymin": int(float(b.find("ymin").text)),
            "xmax": int(float(b.find("xmax").text)),
            "ymax": int(float(b.find("ymax").text)),
        })
    return boxes


def boxes_to_mask_array(boxes: list[dict],
                        tile_size: int, dot_r: int) -> np.ndarray:
    """
    Rasterise bounding boxes as small filled circles (dot_r px radius).
    Returns uint8 array (0 / 255), shape (tile_size, tile_size).
    """
    import cv2
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    for b in boxes:
        xmin = max(0, min(b["xmin"], tile_size - 1))
        ymin = max(0, min(b["ymin"], tile_size - 1))
        xmax = max(0, min(b["xmax"], tile_size - 1))
        ymax = max(0, min(b["ymax"], tile_size - 1))
        cx   = (xmin + xmax) // 2
        cy   = (ymin + ymax) // 2
        cv2.circle(mask, (cx, cy), dot_r, 255, -1)
    return mask


def build_annotation_index(split: str) -> dict[str, str]:
    """
    Returns {tile_stem: xml_path} mapping for one split,
    stripping the Roboflow hash suffix from XML file names.
    """
    annot_dir = Path(ANNOT_ROOTS.get(split, ""))
    if not annot_dir.is_dir():
        print(f"  WARNING  Annotation dir not found: {annot_dir}")
        return {}
    index = {}
    for xml in annot_dir.glob("*.xml"):
        tile_stem = rf_stem_to_tile_stem(xml.stem)
        index[tile_stem] = str(xml)
    return index


def make_masks(tile_meta: list[dict], tile_size: int, dot_r: int):
    """
    For every saved GeoTIFF tile, look up its annotation and write a
    matching binary mask GeoTIFF (1-band uint8, same CRS/transform).
    """
    import cv2   # noqa — used inside boxes_to_mask_array

    # Build per-split annotation indexes
    annot_idx = {}
    for split in ("train", "test"):
        annot_idx[split] = build_annotation_index(split)
        print(f"  Annotation index [{split}]: {len(annot_idx[split])} XMLs")

    stats = {"matched": 0, "missing_xml": 0, "missing_tif": 0}

    for meta in tile_meta:
        split    = meta["split"]
        stem     = meta["stem"]          # e.g. tile_new0_R000_C000
        tile_path = Path(meta["path"])

        if not tile_path.exists():
            stats["missing_tif"] += 1
            continue

        xml_path = annot_idx[split].get(stem)
        if xml_path is None:
            stats["missing_xml"] += 1
            continue

        boxes = parse_voc_boxes(xml_path)
        mask_arr = boxes_to_mask_array(boxes, tile_size, dot_r)

        mask_dir = Path(OUT_MASKS) / split
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{stem}.tif"

        # Copy CRS + transform from the tile
        with rasterio.open(tile_path) as src:
            profile = {
                "driver":    "GTiff",
                "dtype":     "uint8",
                "width":     tile_size,
                "height":    tile_size,
                "count":     1,
                "crs":       src.crs,
                "transform": src.transform,
                "compress":  "lzw",
            }

        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask_arr[np.newaxis, :, :])   # (1, H, W)

        stats["matched"] += 1

    return stats
