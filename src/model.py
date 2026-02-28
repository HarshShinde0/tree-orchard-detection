import torch
import torch.nn as nn

def _cbr(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv(3x3) -> BatchNorm -> ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    """
    Lightweight 3-level U-Net for density-map regression.
    Input : (B, 3, H, W)
    Output: (B, 1, H, W)  - non-negative density values
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = _cbr(in_channels, 32)
        self.enc2 = _cbr(32, 64)
        self.enc3 = _cbr(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _cbr(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _cbr(64, 32)
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.ReLU(),          # ensures density >= 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
