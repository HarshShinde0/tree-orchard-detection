import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults
DEFAULT_ANNOT_DIR  = os.path.join("data", "annotations")
DEFAULT_TILES_DIR  = os.path.join("data", "tiles", "train")
DEFAULT_MASKS_DIR  = os.path.join("data", "masks", "train")
DEFAULT_LABELS_CSV = os.path.join("data", "labels_train.csv")
TILE_SIZE = 512

def parse_args():
    p = argparse.ArgumentParser(description="Prepare dataset from Roboflow VOC exports.")
    p.add_argument("--annotations-dir", default=DEFAULT_ANNOT_DIR)
    p.add_argument("--tiles-dir",       default=DEFAULT_TILES_DIR)
    p.add_argument("--masks-dir",       default=DEFAULT_MASKS_DIR)
    p.add_argument("--labels-csv",      default=DEFAULT_LABELS_CSV)
    p.add_argument("--visualize",       action="store_true")
    return p.parse_args()

def parse_voc_xml(xml_path: str) -> list[dict]:
    """
    Parse a Pascal VOC XML file.
    Returns list of boxes: [{'class': str, 'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        box = {
            "class": cls_name,
            "xmin": int(float(bbox.find("xmin").text)),
            "ymin": int(float(bbox.find("ymin").text)),
            "xmax": int(float(bbox.find("xmax").text)),
            "ymax": int(float(bbox.find("ymax").text)),
        }
        boxes.append(box)

    return boxes

def boxes_to_mask(boxes: list[dict], h: int = TILE_SIZE, w: int = TILE_SIZE) -> np.ndarray:
    """
    Convert bounding boxes to a binary segmentation mask.

    Strategy: Draw a small filled circle at each tree centre using 40% of the
    bounding-box half-width (capped at 20 px).  This keeps individual trees as
    *separate* blobs even in dense orchards, so that:
      - connected-components / watershed can count blobs == tree count
      - U-Net learns to localise individual crowns rather than merged canopy
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        xmin, ymin = box["xmin"], box["ymin"]
        xmax, ymax = box["xmax"], box["ymax"]

        # Clamp to image bounds
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        rx = (xmax - xmin) // 2
        ry = (ymax - ymin) // 2

        # 3 px dot at each tree centre — minimum visible circle.
        # Average tree spacing is ~25 px so 3 px (diameter 6) maximises
        # blob separation in very dense orchards.
        r = 3

        cv2.circle(mask, (cx, cy), r, color=255, thickness=-1)

    return mask

def visualize_overlay(tile_path: str, mask: np.ndarray, boxes: list[dict],
                      out_path: str):
    """Save an overlay visualization: tile + mask + bounding boxes."""
    tile = cv2.imread(tile_path)
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original tile
    axes[0].imshow(tile)
    axes[0].set_title("Tile (RGB)")
    axes[0].axis("off")

    # Mask
    axes[1].imshow(mask, cmap="Greens", vmin=0, vmax=255)
    axes[1].set_title(f"Mask ({len(boxes)} trees)")
    axes[1].axis("off")

    # Overlay
    overlay = tile.copy()
    green_mask = np.zeros_like(tile)
    green_mask[:, :, 1] = mask  # green channel
    overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)

    # Draw bounding boxes
    for box in boxes:
        cv2.rectangle(overlay,
                      (box["xmin"], box["ymin"]),
                      (box["xmax"], box["ymax"]),
                      (255, 255, 0), 1)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    annot_dir = Path(args.annotations_dir).resolve()
    tiles_dir = Path(args.tiles_dir).resolve()
    masks_dir = Path(args.masks_dir).resolve()
    labels_csv = Path(args.labels_csv).resolve()

    #  Validate directories 
    if not annot_dir.is_dir():
        print(f"  ERROR  Annotations directory not found: {annot_dir}")
        print(f"         Export from Roboflow as Pascal VOC and unzip there.")
        sys.exit(1)

    if not tiles_dir.is_dir():
        print(f"  ERROR  Tiles directory not found: {tiles_dir}")
        sys.exit(1)

    masks_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = masks_dir.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Find XML annotation files 
    xml_files = sorted(annot_dir.glob("*.xml"))
    print(f"\n{'='*60}")
    print(f"  Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Annotations dir : {annot_dir}")
    print(f"  Tiles dir       : {tiles_dir}")
    print(f"  Masks dir       : {masks_dir}")
    print(f"  XML files found : {len(xml_files)}")
    print(f"{'='*60}\n")

    if not xml_files:
        print("  ERROR  No .xml annotation files found!")
        print("         Make sure you exported from Roboflow in Pascal VOC format.")
        sys.exit(1)

    # Process each annotation 
    records = []          # for CSV
    total_trees = 0
    processed = 0
    skipped = 0

    for xml_path in xml_files:
        # Find matching tile image
        stem = xml_path.stem
        tile_path = None

        # Try common naming patterns
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = tiles_dir / f"{stem}{ext}"
            if candidate.exists():
                tile_path = candidate
                break

        if tile_path is None:
            print(f"  ⚠  No matching tile for {xml_path.name} — skipping")
            skipped += 1
            continue

        # Parse boxes
        boxes = parse_voc_xml(str(xml_path))
        tree_count = len(boxes)
        total_trees += tree_count

        # Create binary mask (for U-Net)
        mask = boxes_to_mask(boxes)
        mask_path = masks_dir / f"{stem}.png"
        cv2.imwrite(str(mask_path), mask)

        # Record for CSV (for ANN)
        records.append({
            "tile": tile_path.name,
            "xml": xml_path.name,
            "tree_count": tree_count,
        })

        processed += 1

        if processed % 25 == 0 or processed == 1:
            print(f"  {processed:>3}/{len(xml_files)}  {stem}  →  {tree_count} trees")

        # Optional visualization
        if args.visualize and processed <= 20:
            vis_path = vis_dir / f"vis_{stem}.png"
            visualize_overlay(str(tile_path), mask, boxes, str(vis_path))

    # Save labels CSV
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile", "xml", "tree_count"])
        writer.writeheader()
        writer.writerows(records)

    # Summary 
    print(f"\n{'='*60}")
    print(f"  Dataset Preparation Complete!")
    print(f"{'='*60}")
    print(f"  Processed       : {processed} tiles")
    print(f"  Skipped         : {skipped}")
    print(f"  Total trees     : {total_trees}")
    if processed > 0:
        print(f"  Avg trees/tile  : {total_trees / processed:.1f}")
    print(f"\n  Outputs:")
    print(f"    U-Net masks   : {masks_dir}/ ({processed} .png files)")
    print(f"    ANN labels    : {labels_csv}")
    if args.visualize:
        print(f"    Visualizations: {vis_dir}/")
    print(f"\n  Next: Run 04_train_unet.py or 04_train_ann.py")

if __name__ == "__main__":
    main()
