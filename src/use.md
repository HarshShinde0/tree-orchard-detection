# Tree / Orchard detection counting from high resolution satellite images
This repository contains code to perform density map regression with U-Net (PyTorch) for tree / orchard detection counting from high resolution satellite images.

## Structure
- `config.yml`: Adjust configurations, hyperparameters, and directory paths. You can modify paths like `train_img`, `train_mask`, and `target_image`.
- `train.py`: Run this script to train the model over the dataset.
- `predict.py`: Run this script to perform inference on a single test image and calculate the prediction heatmap.
- `requirements.txt`: Python dependencies needed to run the project.

## Installation
Use standard `pip` environments:
```bash
pip install -r requirements.txt
```

## Usage
1. Modify `config.yml` according to your data locations. Ensure the paths specified exist. By default, it expects a dataset folder parallel to the code directory (`../dataset`).
2. Run `cd src && python train.py` to train your model. This saves `best_model.pth`.
3. Run `cd src && python predict.py` to run model inference on a target image and generate overlays.

