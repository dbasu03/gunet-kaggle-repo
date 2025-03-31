# models.py (updated)
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class GUNet(nn.Module):
    def __init__(self):
        super(GUNet, self).__init__()
        self.patch_size = 16  # For padding purposes

        # Encoder
        self.enc1 = ConvBlock(3, 64)    # Input: 3, Output: 64
        self.enc2 = ConvBlock(64, 128)  # Input: 64, Output: 128
        self.enc3 = ConvBlock(128, 256) # Input: 128, Output: 256
        self.enc4 = ConvBlock(256, 512) # Input: 256, Output: 512

        # Downsampling layers with matching channels
        self.down1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)   # After enc1: 64 -> 64
        self.down2 = nn.Conv2d(128, 128, kernel_size=2, stride=2) # After enc2: 128 -> 128
        self.down3 = nn.Conv2d(256, 256, kernel_size=2, stride=2) # After enc3: 256 -> 256

        # Skip connections with fusion
        self.fusion1 = nn.Conv2d(64 + 64, 64, kernel_size=1)   # Skip connection: 64 (from up3) + 64 (from enc1) -> 64
        self.fusion2 = nn.Conv2d(128 + 128, 128, kernel_size=1) # Skip connection: 128 (from up2) + 128 (from enc2) -> 128
        self.fusion3 = nn.Conv2d(256 + 256, 256, kernel_size=1) # Skip connection: 256 (from up1) + 256 (from enc3) -> 256

        # Decoder
        self.dec3 = ConvBlock(256, 256) # Input: 256 (from fusion3) -> 256
        self.dec2 = ConvBlock(128, 128) # Input: 128 (from fusion2) -> 128
        self.dec1 = ConvBlock(64, 64)   # Input: 64 (from fusion1) -> 64
        self.out = nn.Conv2d(64, 3, kernel_size=1) # Final output: 64 -> 3

        # Upsampling layers with matching channels
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # After enc4: 512 -> 256
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # After dec3: 256 -> 128
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # After dec2: 128 -> 64

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # Shape: [B, 64, H, W]
        d1 = self.down1(e1)  # Shape: [B, 64, H/2, W/2]
        e2 = self.enc2(d1)   # Shape: [B, 128, H/2, W/2]
        d2 = self.down2(e2)  # Shape: [B, 128, H/4, W/4]
        e3 = self.enc3(d2)   # Shape: [B, 256, H/4, W/4]
        d3 = self.down3(e3)  # Shape: [B, 256, H/8, W/8]
        e4 = self.enc4(d3)   # Shape: [B, 512, H/8, W/8]

        # Decoder with skip connections
        u1 = self.up1(e4)    # Shape: [B, 256, H/4, W/4]
        f3 = self.fusion3(torch.cat([u1, e3], dim=1)) # Shape: [B, 256, H/4, W/4]
        d3 = self.dec3(f3)   # Shape: [B, 256, H/4, W/4]

        u2 = self.up2(d3)    # Shape: [B, 128, H/2, W/2]
        f2 = self.fusion2(torch.cat([u2, e2], dim=1)) # Shape: [B, 128, H/2, W/2]
        d2 = self.dec2(f2)   # Shape: [B, 128, H/2, W/2]

        u3 = self.up3(d2)    # Shape: [B, 64, H, W]
        f1 = self.fusion1(torch.cat([u3, e1], dim=1)) # Shape: [B, 64, H, W]
        d1 = self.dec1(f1)   # Shape: [B, 64, H, W]

        out = self.out(d1)   # Shape: [B, 3, H, W]
        return out

def gunet_t():
    return GUNet()
