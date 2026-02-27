import torch
import torch.nn.functional as F

EPSILON = 1e-5

def _split(y_true):
    """y_true (B, 2, H, W) → mask (B,1,H,W), weights (B,1,H,W)"""
    return y_true[:, 0:1, :, :], y_true[:, 1:2, :, :]

# training loss 
def bce_dice_loss(y_true, y_pred, bce_weight=0.5):
    """
    BCE + Dice combined loss.
    Dice prevents collapse to all-background on imbalanced tree masks.
    y_true : (B, 2, H, W) – channel 0 = mask, channel 1 = weight map
    y_pred : (B, 1, H, W)
    """
    y_t, y_w = _split(y_true)
    # weighted BCE
    bce = F.binary_cross_entropy(y_pred, y_t, reduction='none')
    bce_val = (bce * y_w).mean(dim=[0, 2, 3]).sum()
    # Dice
    smooth = 1e-7
    inter  = torch.sum(torch.abs(y_t * y_pred), dim=[2, 3])
    union  = torch.sum(y_t, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
    dice_val = 1.0 - torch.mean((2.0 * inter + smooth) / (union + smooth))
    return bce_weight * bce_val + (1.0 - bce_weight) * dice_val

# metrics
def accuracy(y_true, y_pred):
    y_t, _ = _split(y_true)
    return (torch.round(y_t) == torch.round(y_pred)).float().mean()

def precision(y_true, y_pred):
    y_t, _ = _split(y_true)
    tp = torch.round(y_t * y_pred).sum()
    fp = torch.round((1.0 - y_t) * y_pred).sum()
    return tp / (tp + fp + EPSILON)

def recall(y_true, y_pred):
    y_t, _ = _split(y_true)
    tp = torch.round(y_t * y_pred).sum()
    fn = torch.round(y_t * (1.0 - y_pred)).sum()
    return tp / (tp + fn + EPSILON)
