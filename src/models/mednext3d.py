import torch
import torch.nn as nn
import torch.nn.functional as F

class MedNeXtBlock3D(nn.Module):
    """
    3D MedNeXt / ConvNeXt Inverted Bottleneck Block with Large Depthwise Convolutions.
    Uses large kernel (5x5x5 or 7x7x7) for massive 3D anatomical receptive field.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, expansion_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2
        
        # 1. Large 3D Depthwise Convolution (Spatial filtering per channel)
        self.dw_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=False
        )
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)
        
        # 2. Inverted Bottleneck Pointwise 1x1x1 Conv (Channel expansion)
        expanded_channels = in_channels * expansion_ratio
        self.pw_conv1 = nn.Conv3d(in_channels, expanded_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        
        # 3. Pointwise 1x1x1 Conv (Channel projection)
        self.pw_conv2 = nn.Conv3d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.norm(self.dw_conv(x))
        out = self.act(self.pw_conv1(out))
        out = self.norm2(self.pw_conv2(out))
        return out + res

class MedNeXtDownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.block = MedNeXtBlock3D(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.down_conv(x))
        return self.block(out)

class MedNeXtUpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.block = MedNeXtBlock3D(out_channels + skip_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.upconv(x))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class MedNeXt3D(nn.Module):
    """
    MedNeXt-3D Architecture tailored for mandibular canal segmentation.
    Uses large 5x5x5/7x7x7 depthwise separable convolutions to capture continuous curvilinear tracts.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3, feature_dims: list = None, kernel_size: int = 5, deep_supervision: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        dims = feature_dims or [24, 48, 96, 192, 256]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(dims[0], affine=True),
            nn.GELU(),
            MedNeXtBlock3D(dims[0], dims[0], kernel_size=kernel_size)
        )
        
        # Encoder
        self.enc1 = MedNeXtDownBlock3D(dims[0], dims[1], kernel_size=kernel_size)
        self.enc2 = MedNeXtDownBlock3D(dims[1], dims[2], kernel_size=kernel_size)
        self.enc3 = MedNeXtDownBlock3D(dims[2], dims[3], kernel_size=kernel_size)
        self.bottleneck = MedNeXtDownBlock3D(dims[3], dims[4], kernel_size=kernel_size)
        
        # Decoder
        self.dec3 = MedNeXtUpBlock3D(dims[4], dims[3], dims[3], kernel_size=kernel_size)
        self.dec2 = MedNeXtUpBlock3D(dims[3], dims[2], dims[2], kernel_size=kernel_size)
        self.dec1 = MedNeXtUpBlock3D(dims[2], dims[1], dims[1], kernel_size=kernel_size)
        self.dec0 = MedNeXtUpBlock3D(dims[1], dims[0], dims[0], kernel_size=kernel_size)
        
        # Output Heads
        self.final_head = nn.Conv3d(dims[0], num_classes, kernel_size=1)
        
        if self.deep_supervision:
            self.ds_head1 = nn.Conv3d(dims[1], num_classes, kernel_size=1)
            self.ds_head2 = nn.Conv3d(dims[2], num_classes, kernel_size=1)
            self.ds_head3 = nn.Conv3d(dims[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # Stem & Encoder
        e0 = self.stem(x)
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
            target_size = x.shape[2:]
            out1 = F.interpolate(self.ds_head1(d1), size=target_size, mode='trilinear', align_corners=False)
            out2 = F.interpolate(self.ds_head2(d2), size=target_size, mode='trilinear', align_corners=False)
            out3 = F.interpolate(self.ds_head3(d3), size=target_size, mode='trilinear', align_corners=False)
            return [out0, out1, out2, out3]
            
        return out0
