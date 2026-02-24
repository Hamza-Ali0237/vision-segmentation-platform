import torch
import torch.nn as nn

from training.models.encoder import Encoder
from decoder import Decoder
from bottleneck import Bottleneck

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = Encoder(in_channels, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        # Bottleneck
        self.bottleneck = Bottleneck()

        # Decoder
        self.dec4 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec2 = Decoder(128, 64)
        self.dec1 = Decoder(64, out_channels)

        # Final Conv 1x1
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1_out, enc1_pool = self.enc1(x)
        enc2_out, enc2_pool = self.enc2(enc1_pool)
        enc3_out, enc3_pool = self.enc3(enc2_pool)
        enc4_out, enc4_pool = self.enc4(enc3_pool)

        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_pool)

        # Decoder
        dec4_out = self.dec4(bottleneck_out, enc4_out)
        dec3_out = self.dec3(dec4_out, enc3_out)
        dec2_out = self.dec2(dec3_out, enc2_out)
        dec1_out = self.dec1(dec2_out, enc1_out)

        # Final Conv
        final_output = self.final_conv(dec1_out)

        return final_output