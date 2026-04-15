import torch
import torch.nn as nn

from src.models.UNet.encoder import Encoder
from src.models.UNet.decoder import Decoder
from src.models.UNet.bottleneck import Bottleneck


class UNet(nn.Module):
    """
    Standard UNet for binary segmentation.
    Input:  (B, 3, H, W)
    Output: (B, 1, H', W')  — raw logits, sigmoid applied externally.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder path
        self.enc1 = Encoder(in_channels, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        # Bridge
        self.bottleneck = Bottleneck()

        # Decoder path — in_channels = bottleneck/prev output channels
        self.dec4 = Decoder(1024, 512)
        self.dec3 = Decoder(512,  256)
        self.dec2 = Decoder(256,  128)
        self.dec1 = Decoder(128,  64)

        # 1×1 projection to desired output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1_feat, enc1_pool = self.enc1(x)
        enc2_feat, enc2_pool = self.enc2(enc1_pool)
        enc3_feat, enc3_pool = self.enc3(enc2_pool)
        enc4_feat, enc4_pool = self.enc4(enc3_pool)

        # Bottleneck
        bn = self.bottleneck(enc4_pool)

        # Decoder (each block gets the corresponding skip connection)
        d4 = self.dec4(bn,  enc4_feat)
        d3 = self.dec3(d4,  enc3_feat)
        d2 = self.dec2(d3,  enc2_feat)
        d1 = self.dec1(d2,  enc1_feat)

        return self.final_conv(d1)