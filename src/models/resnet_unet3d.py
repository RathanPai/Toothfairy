import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu2(out + res)
        return out

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.res_block = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        # Handle small spatial dimension mismatch if odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)

class ResEncoderUNet3D(nn.Module):
    """
    3D Residual U-Net with deep supervision and Instance Normalization.
    Optimized for GPU memory efficiency and rich multi-scale feature representation.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3, feature_dims: list = None, deep_supervision: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        dims = feature_dims or [24, 48, 96, 192, 256]
        
        # Encoder Stages
        self.enc0 = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(dims[0], affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            ResidualBlock3D(dims[0], dims[0])
        )
        self.enc1 = nn.Sequential(
            ResidualBlock3D(dims[0], dims[1], stride=2),
            ResidualBlock3D(dims[1], dims[1])
        )
        self.enc2 = nn.Sequential(
            ResidualBlock3D(dims[1], dims[2], stride=2),
            ResidualBlock3D(dims[2], dims[2])
        )
        self.enc3 = nn.Sequential(
            ResidualBlock3D(dims[2], dims[3], stride=2),
            ResidualBlock3D(dims[3], dims[3])
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(dims[3], dims[4], stride=2),
            ResidualBlock3D(dims[4], dims[4])
        )
        
        # Decoder Stages
        self.dec3 = DecoderBlock3D(dims[4], dims[3], dims[3])
        self.dec2 = DecoderBlock3D(dims[3], dims[2], dims[2])
        self.dec1 = DecoderBlock3D(dims[2], dims[1], dims[1])
        self.dec0 = DecoderBlock3D(dims[1], dims[0], dims[0])
        
        # Output Heads
        self.final_head = nn.Conv3d(dims[0], num_classes, kernel_size=1)
        
        # Deep Supervision Heads
        if self.deep_supervision:
            self.ds_head1 = nn.Conv3d(dims[1], num_classes, kernel_size=1)
            self.ds_head2 = nn.Conv3d(dims[2], num_classes, kernel_size=1)
            self.ds_head3 = nn.Conv3d(dims[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        
        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        
        out0 = self.final_head(d0)
        
        if self.training and self.deep_supervision:
            # Interpolate deep supervision outputs to full input size
            target_size = x.shape[2:]
            out1 = F.interpolate(self.ds_head1(d1), size=target_size, mode='trilinear', align_corners=False)
            out2 = F.interpolate(self.ds_head2(d2), size=target_size, mode='trilinear', align_corners=False)
            out3 = F.interpolate(self.ds_head3(d3), size=target_size, mode='trilinear', align_corners=False)
            return [out0, out1, out2, out3]
            
        return out0
