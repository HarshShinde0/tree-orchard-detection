import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from losses import bce_dice_loss, accuracy, precision, recall

# Building blocks
class DoubleConv(nn.Module):
    """Two consecutive Conv2d-ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# U-Net
class UNet(nn.Module):
    """
    U-Net with 5 encoder blocks and 5 decoder blocks.

    Input  shape : (B, n_channels, im_sz, im_sz)
    Output shape : (B, n_classes,  im_sz, im_sz)  – values in (0, 1) via sigmoid
    """

    def __init__(
        self,
        n_classes: int = 1,
        im_sz: int = 160,
        n_channels: int = 4,
        n_filters_start: int = 32,
        growth_factor: int = 2,
        upconv: bool = True,
        droprate: float = 0.25,
    ):
        super().__init__()
        self.upconv = upconv
        g = growth_factor
        f1 = n_filters_start          # 32
        f2 = f1 * g                   # 64
        f3 = f2 * g                   # 128
        f4 = f3 * g                   # 256
        f5 = f4 * g                   # 512
        f6 = f5 * g                   # 1024  (bottleneck)

        # Encoder 
        self.enc1    = DoubleConv(n_channels, f1)
        self.pool1   = nn.MaxPool2d(2)

        self.bn_e2   = nn.BatchNorm2d(f1)
        self.enc2    = DoubleConv(f1, f2)
        self.pool2   = nn.MaxPool2d(2)
        self.drop2   = nn.Dropout2d(droprate)

        self.bn_e3   = nn.BatchNorm2d(f2)
        self.enc3    = DoubleConv(f2, f3)
        self.pool3   = nn.MaxPool2d(2)
        self.drop3   = nn.Dropout2d(droprate)

        self.bn_e4   = nn.BatchNorm2d(f3)
        self.enc4_0  = DoubleConv(f3, f4)
        self.pool4_1 = nn.MaxPool2d(2)
        self.drop4_1 = nn.Dropout2d(droprate)

        self.bn_e5   = nn.BatchNorm2d(f4)
        self.enc4_1  = DoubleConv(f4, f5)
        self.pool4_2 = nn.MaxPool2d(2)
        self.drop4_2 = nn.Dropout2d(droprate)

        # Bottleneck
        self.bottleneck = DoubleConv(f5, f6)

        # Decoder 
        # When upconv=True transposed convolution halves channels before concat.
        # When upconv=False bilinear upsample keeps channels, so concat is larger.
        d6_1_in = (f5 + f5) if upconv else (f6 + f5)   # 1024 or 1536
        d6_2_in = (f4 + f4) if upconv else (f5 + f4)   # 512  or 768
        d7_in   = (f3 + f3) if upconv else (f4 + f3)   # 256  or 384
        d8_in   = (f2 + f2) if upconv else (f3 + f2)   # 128  or 192
        d9_in   = (f1 + f1) if upconv else (f2 + f1)   # 64   or 96

        if upconv:
            self.up6_1 = nn.ConvTranspose2d(f6, f5, kernel_size=2, stride=2)
            self.up6_2 = nn.ConvTranspose2d(f5, f4, kernel_size=2, stride=2)
            self.up7   = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
            self.up8   = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
            self.up9   = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.bn_d6_1  = nn.BatchNorm2d(d6_1_in)
        self.dec6_1   = DoubleConv(d6_1_in, f5)
        self.drop6_1  = nn.Dropout2d(droprate)

        self.bn_d6_2  = nn.BatchNorm2d(d6_2_in)
        self.dec6_2   = DoubleConv(d6_2_in, f4)
        self.drop6_2  = nn.Dropout2d(droprate)

        self.bn_d7    = nn.BatchNorm2d(d7_in)
        self.dec7     = DoubleConv(d7_in, f3)
        self.drop7    = nn.Dropout2d(droprate)

        self.bn_d8    = nn.BatchNorm2d(d8_in)
        self.dec8     = DoubleConv(d8_in, f2)
        self.drop8    = nn.Dropout2d(droprate)

        # Note: last skip-merge has no BN/dropout in the original
        self.dec9     = DoubleConv(d9_in, f1)

        self.out_conv = nn.Conv2d(f1, n_classes, kernel_size=1)

    def _up(self, x: torch.Tensor, skip: torch.Tensor, up_layer) -> torch.Tensor:
        if self.upconv:
            x = up_layer(x)
        else:
            x = self.upsample(x)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        p1 = self.bn_e2(p1)
        s2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(s2))

        p2 = self.bn_e3(p2)
        s3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(s3))

        p3 = self.bn_e4(p3)
        s4_0 = self.enc4_0(p3)
        p4_1 = self.drop4_1(self.pool4_1(s4_0))

        p4_1 = self.bn_e5(p4_1)
        s4_1 = self.enc4_1(p4_1)
        p4_2 = self.drop4_2(self.pool4_2(s4_1))

        # Bottleneck
        b = self.bottleneck(p4_2)

        # Decoder
        d = self._up(b,    s4_1, self.up6_1 if self.upconv else None)
        d = self.bn_d6_1(d);  d = self.drop6_1(self.dec6_1(d))

        d = self._up(d,    s4_0, self.up6_2 if self.upconv else None)
        d = self.bn_d6_2(d);  d = self.drop6_2(self.dec6_2(d))

        d = self._up(d,    s3,   self.up7   if self.upconv else None)
        d = self.bn_d7(d);    d = self.drop7(self.dec7(d))

        d = self._up(d,    s2,   self.up8   if self.upconv else None)
        d = self.bn_d8(d);    d = self.drop8(self.dec8(d))

        d = self._up(d,    s1,   self.up9   if self.upconv else None)
        d = self.dec9(d)

        return torch.sigmoid(self.out_conv(d))

# Factory / convenience helpers
def unet_model(
    n_classes: int = 1,
    im_sz: int = 160,
    n_channels: int = 4,
    n_filters_start: int = 32,
    growth_factor: int = 2,
    upconv: bool = True,
    droprate: float = 0.25,
) -> UNet:
    """Build and return a UNet instance (matches original API)."""
    return UNet(
        n_classes=n_classes,
        im_sz=im_sz,
        n_channels=n_channels,
        n_filters_start=n_filters_start,
        growth_factor=growth_factor,
        upconv=upconv,
        droprate=droprate,
    )

def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Adam:
    """Return an Adam optimiser bound to *model*'s parameters."""
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def model_load(model_path: str, **kwargs) -> UNet:
    """
    Load a previously saved UNet checkpoint.

    Saves are expected to be produced by:
        torch.save({'model_state': model.state_dict(), 'kwargs': ...}, path)
    or a bare state-dict file:
        torch.save(model.state_dict(), path)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        ctor_kwargs = {**checkpoint.get('kwargs', {}), **kwargs}
        model = UNet(**ctor_kwargs)
        model.load_state_dict(checkpoint['model_state'])
    else:
        # bare state-dict – caller must pass constructor kwargs
        model = UNet(**kwargs)
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded from", model_path)
    return model

# Training step & metrics 
def train_step(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
):
    """
    Single supervised training step.

    x : (B, n_channels, H, W)
    y : (B, 2*n_classes, H, W) – first half masks, second half weights
    """
    model.train()
    x, y = x.to(device), y.to(device)
    optimiser.zero_grad()
    pred = model(x)
    loss = bce_dice_loss(y, pred)
    loss.backward()
    optimiser.step()

    with torch.no_grad():
        acc  = accuracy(y, pred).item()
        prec = precision(y, pred).item()
        rec  = recall(y, pred).item()

    return {'loss': loss.item(), 'accuracy': acc, 'precision': prec, 'recall': rec}

@torch.no_grad()
def val_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
):
    """Single validation step (no gradient)."""
    model.eval()
    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = bce_dice_loss(y, pred)
    acc  = accuracy(y, pred).item()
    prec = precision(y, pred).item()
    rec  = recall(y, pred).item()
    return {'val_loss': loss.item(), 'val_accuracy': acc, 'val_precision': prec, 'val_recall': rec}
