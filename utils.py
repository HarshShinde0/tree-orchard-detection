import os
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage import morphology, feature, segmentation
from skimage import filters
from scipy import ndimage as ndi
import torch
from torch.utils.data import Dataset, DataLoader

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

def get_rand_patch(img, mask, sz=160, channel=None):
    """
    img  : (H, W, C)  normalised float
    mask : (H, W, 1)  binary float [0, 1]
    Returns a randomly cropped + augmented patch pair.
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass

    # Random brightness/contrast jitter on image only (values are in [-1, 1])
    scale = np.random.uniform(0.85, 1.15)
    shift = np.random.uniform(-0.08, 0.08)
    patch_img = np.clip(patch_img * scale + shift, -1.0, 1.0)

    # Random Gaussian noise
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.02, patch_img.shape).astype(np.float32)
        patch_img = np.clip(patch_img + noise, -1.0, 1.0)

    if channel=='all':
        return patch_img, patch_mask
    
    if channel !='all':
        patch_mask = patch_mask[:,:,channel]
        return patch_img, patch_mask

def get_patches(x_dict, y_dict, n_patches, sz=160, channel='all'):
    """
    Sample random patches from the image/mask dictionaries.

    Returns
    -------
    x : torch.Tensor  (B, C, H, W)  float32, normalised
    y : torch.Tensor  (B, n_classes, H, W)  float32
                      or (B, 1, H, W) when a single channel is selected

    Channels  0: Buildings, 1: Roads & Tracks, 2: Trees, 3: Crops, 4: Water
    """
    x_list, y_list = [], []
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(list(x_dict.keys()), 1)[0]
        img_patch, mask_patch = get_rand_patch(x_dict[img_id], y_dict[img_id], sz, channel)
        x_list.append(img_patch)
        y_list.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))

    # numpy (B, H, W, C)  →  torch (B, C, H, W)
    x_np = np.array(x_list, dtype=np.float32)           # (B, H, W, n_ch)
    y_np = np.array(y_list, dtype=np.float32)           # (B, H, W, n_cls) or (B, H, W)

    x_t = torch.from_numpy(x_np).permute(0, 3, 1, 2)   # (B, n_ch, H, W)

    if y_np.ndim == 3:                                  # single channel selected
        y_t = torch.from_numpy(y_np).unsqueeze(1)       # (B, 1, H, W)
    else:
        y_t = torch.from_numpy(y_np).permute(0, 3, 1, 2)  # (B, n_cls, H, W)

    return x_t, y_t

def load_data(path='./dataset/'):
    """
    Load train and test tiles from dataset/image/{train,test}/ and
    dataset/annotation/{train,test}/ using rasterio.

    Images  : uint16 (4, H, W) -> float32 (H, W, 4), normalised to [-1, 1]
    Masks   : uint8  (1, H, W), values 0/255 -> float32 (H, W, 1), values [0, 1]

    Returns X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_TEST, Y_DICT_TEST
    """
    result = {}
    for split in ('train', 'test'):
        img_dir = os.path.join(path, 'image', split)
        ann_dir = os.path.join(path, 'annotation', split)
        fnames  = sorted(os.listdir(img_dir))
        x_dict, y_dict = {}, {}
        print(f'Reading {split} images ({len(fnames)} tiles)...')
        for fname in fnames:
            tile_id = os.path.splitext(fname)[0]
            with rasterio.open(os.path.join(img_dir, fname)) as src:
                img = src.read().astype(np.float32)   # (C, H, W)
            with rasterio.open(os.path.join(ann_dir, fname)) as src:
                ann = src.read().astype(np.float32)   # (1, H, W), values 0/255
            img = normalize(img.transpose(1, 2, 0))   # (H, W, C), [-1, 1]
            ann = (ann / 255.0).transpose(1, 2, 0)    # (H, W, 1), [0, 1]
            x_dict[tile_id] = img
            y_dict[tile_id] = ann
        result[split] = (x_dict, y_dict)
    print('Images are read')
    X_DICT_TRAIN, Y_DICT_TRAIN = result['train']
    X_DICT_TEST,  Y_DICT_TEST  = result['test']
    return X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_TEST, Y_DICT_TEST

# PyTorch Dataset
def masks_to_weighted_target(mask_t: torch.Tensor) -> torch.Tensor:
    """
    Build the stacked y_true tensor expected by losses.py.

    mask_t : (C, H, W)  float32, values in [0, 1]
    Returns : (2*C, H, W)  – first C channels are the masks,
                             last C channels are uniform weight maps (1.0)
    """
    weights = torch.ones_like(mask_t)
    return torch.cat([mask_t, weights], dim=0)

class PatchDataset(Dataset):
    """
    Torch Dataset that samples random patches on-the-fly.

    Parameters
    ----------
    x_dict : dict {id -> ndarray (H, W, n_channels)}  normalised images
    y_dict : dict {id -> ndarray (H, W, n_classes)}   binary masks [0, 1]
    n_patches : total number of patches to serve per epoch
    sz        : patch size (square)
    channel   : int or 'all'  – which mask channel(s) to use
    weighted  : if True, stack uniform weight maps → y shape (2*C, H, W)
                so it can be fed directly into losses.weighted_binary_crossentropy
    """

    def __init__(self, x_dict, y_dict, n_patches=1000, sz=160,
                 channel='all', weighted=True):
        self.x_dict    = x_dict
        self.y_dict    = y_dict
        self.n_patches = n_patches
        self.sz        = sz
        self.channel   = channel
        self.weighted  = weighted

        # pre-sample all patches once per epoch
        self._resample()

    def _resample(self):
        """Draw a fresh set of patches (call at start of each epoch if desired)."""
        self.x_patches, self.y_patches = get_patches(
            self.x_dict, self.y_dict, self.n_patches, self.sz, self.channel
        )   # (B, C, H, W) tensors

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        x = self.x_patches[idx]   # (n_channels, H, W)
        y = self.y_patches[idx]   # (n_classes,  H, W)  or  (1, H, W)
        if self.weighted:
            y = masks_to_weighted_target(y)   # (2*n_classes, H, W)
        return x, y

def make_dataloaders(x_train, y_train, x_val, y_val,
                     n_patches_train=2000, n_patches_val=500,
                     sz=160, channel='all', batch_size=16,
                     num_workers=0, weighted=True):
    """
    Convenience factory that returns (train_loader, val_loader).

    Usage
    -----
        X_TR, Y_TR, X_VA, Y_VA = load_data()
        train_dl, val_dl = make_dataloaders(X_TR, Y_TR, X_VA, Y_VA)
        for x, y in train_dl:
            metrics = train_step(model, optimiser, x, y, device)
    """
    train_ds = PatchDataset(x_train, y_train, n_patches_train, sz, channel, weighted)
    val_ds   = PatchDataset(x_val,   y_val,   n_patches_val,   sz, channel, weighted)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl

# Post-processing helpers (operate on numpy arrays; call after .cpu().numpy())
def plot_train_data(X_DICT_TRAIN, Y_DICT_TRAIN, image_number=12):
    if isinstance(image_number, int):
        image_number = sorted(X_DICT_TRAIN.keys())[image_number]
    img  = X_DICT_TRAIN[image_number]   # (H, W, 4)
    mask = Y_DICT_TRAIN[image_number]   # (H, W, 1)
    rgb  = img[:, :, :3]
    rgb  = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(rgb);            ax1.set_title('Image (RGB preview)'); ax1.axis('off')
    ax2.imshow(mask[:, :, 0], cmap='gray'); ax2.set_title('Ground Truth: Trees'); ax2.axis('off')
    plt.tight_layout()
    plt.show()

# helpers used by post_processing 

def Gaussian_filter(image, sigma=1):
    return filters.gaussian(np.copy(image), sigma=sigma)

def Find_threshold_otsu(image):
    return filters.threshold_otsu(image)

def Binary(image, threshold, max_value=1):
    img = np.copy(image).astype(np.float32)
    return (img > threshold).astype(np.float32) * max_value

def post_processing(img, min_tree_size=20, min_distance=8):
    """
    Convert a raw UNet probability map into individual tree instances.
    binary_mask      : 2-D uint8 – clean binary foreground mask
    n_trees          : int – number of individual trees detected
    instance_labels  : 2-D int – each tree has a unique integer label (0 = bg)
    """
    # 1. Smooth + threshold
    blur       = Gaussian_filter(img, sigma=1)
    t          = Find_threshold_otsu(blur)
    binary_mask = Binary(blur, t).astype(np.uint8)

    # 2. Remove small noise blobs
    binary_mask = morphology.remove_small_objects(
        binary_mask.astype(bool), min_size=min_tree_size
    ).astype(np.uint8)

    # 3. Distance transform — peaks = tree centres
    dist   = ndi.distance_transform_edt(binary_mask)

    # 4. Find local peaks (one per tree)
    coords = feature.peak_local_max(
        dist,
        min_distance=min_distance,
        labels=binary_mask,
    )
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    # 5. Watershed — splits touching canopies using the distance map as terrain
    instance_labels = segmentation.watershed(-dist, markers, mask=binary_mask)

    # 6. Count unique trees (exclude background label 0)
    n_trees = len(np.unique(instance_labels)) - 1

    return binary_mask, n_trees, instance_labels

def count_trees(pred_mask, min_tree_size=20, min_distance=8):
    """
    Convenience wrapper: takes a UNet output tensor or numpy array,
    runs post_processing, and returns the tree count + labelled instance map.

    pred_mask : (1, H, W) or (H, W) tensor/ndarray, values in [0, 1]
    """
    if hasattr(pred_mask, 'cpu'):          # torch tensor
        pred_mask = pred_mask.detach().cpu().numpy()
    pred_mask = np.squeeze(pred_mask)      # → (H, W)
    binary_mask, n_trees, instance_labels = post_processing(
        pred_mask, min_tree_size=min_tree_size, min_distance=min_distance
    )
    return n_trees, instance_labels, binary_mask    
