import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        # After cat: out_channels (upsampled) + out_channels (skip) = out_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, enc_feat: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        # Pad x if there's a 1-pixel mismatch from odd input dimensions
        if x.shape != enc_feat.shape:
            x = nn.functional.pad(x, [0, enc_feat.shape[3] - x.shape[3],
                                       0, enc_feat.shape[2] - x.shape[2]])
        x = torch.cat([enc_feat, x], dim=1)
        return self.conv(x)