import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import load_config
from model import UNet
from utils import read_tif, tensor_to_rgb

def crop_to_multiple(img, multiple=4):
    c, h, w = img.shape
    h = h - (h % multiple)
    w = w - (w % multiple)
    return img[:, :h, :w]

def make_overlay(rgb, dens):
    threshold = 0.1 * dens.max()
    mask = dens > threshold

    overlay = rgb.copy()

    if np.any(mask):
        norm = (dens - threshold) / (dens.max() - threshold + 1e-6)
        norm = np.clip(norm, 0, 1)

        # Apply red overlay
        overlay[mask, 0] = overlay[mask, 0] * 0.3 + (0.8 + 0.2 * norm[mask])
        overlay[mask, 1] *= 0.3
        overlay[mask, 2] *= 0.3

    return np.clip(overlay, 0, 1)

def infer_and_plot_on_image(image_path, model_path, device):
    print(f"Running inference on {image_path} with model {model_path}")
    model = UNet(in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = read_tif(image_path)
    img = crop_to_multiple(img, 4)
    rgb = tensor_to_rgb(img)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()

    overlay = make_overlay(rgb, pred)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Original Image", fontsize=12)
    ax[0].axis("off")
    ax[0].set_aspect("equal")

    im = ax[1].imshow(pred, cmap="jet")
    ax[1].set_title(f"Predicted Density\nSum={pred.sum():.1f}", fontsize=12)
    ax[1].axis("off")
    ax[1].set_aspect("equal")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay", fontsize=12)
    ax[2].axis("off")
    ax[2].set_aspect("equal")

    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("inference_output.png")
    print("Saved output figure to inference_output.png")
    plt.show()

    print("Total predicted count:", pred.sum())

def main():
    config = load_config("../config.yml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def resolve_path(p):
        return os.path.normpath(os.path.join(base_dir, p.lstrip("./\\")))

    image_path = resolve_path(config['inference']['target_image'])
    model_path = resolve_path(config['training']['model_save_path'])
    
    infer_and_plot_on_image(image_path, model_path, device)

if __name__ == '__main__':
    main()
