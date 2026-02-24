import torch
import torch.nn as nn

def center_crop(enc_feat, target_tensor):
    _, _, h, w = target_tensor.shape
    enc_h, enc_w = enc_feat.shape[2], enc_feat.shape[3]

    delta_h = enc_h - h
    delta_w = enc_w - w

    return enc_feat[
        :, :,
        delta_h // 2 : enc_h - delta_h // 2,
        delta_w // 2 : enc_w - delta_w // 2
    ]



class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, enc_feat):
        x = self.up_conv(x)
        enc_feat = center_crop(enc_feat, x)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.conv(x)
        return x