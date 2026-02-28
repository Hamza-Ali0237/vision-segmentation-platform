import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        # Note: This is a smaller version of SegNet, considering the size of the dataset is quite small too.

        # Encoder
        num_conv_layers_enc = [2, 3, 3] # Number of Conv2d layers per encoder block

        self.enc1 = Encoder(3, 64, num_conv_layers_enc[0])
        self.enc2 = Encoder(64, 128, num_conv_layers_enc[1])
        self.enc3 = Encoder(128, 128, num_conv_layers_enc[2])

        # Decoder
        num_conv_layers_dec = [3, 3, 2] # Number of Conv2d layers per decoder block

        self.dec3 = Decoder(128, 128, num_conv_layers_dec[0])
        self.dec2 = Decoder(128, 64, num_conv_layers_dec[1])
        self.dec1 = Decoder(64, num_classes, num_conv_layers_dec[2])



    def forward(self, x):
        x, ind1 = self.enc1(x)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)

        x = self.dec3(x, ind3)
        x = self.dec2(x, ind2)
        x = self.dec1(x, ind1)

        return x