import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(Encoder, self).__init__()

        self.enc_block = nn.ModuleList()

        current_in_channels = in_channels

        for _ in range(num_conv_layers):
            layer = nn.Sequential(
                nn.Conv2d(current_in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.enc_block.append(layer)

            current_in_channels = out_channels

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    
    def forward(self, x):
        for layer in self.enc_block:
            x = layer(x)
        
        x, indices = self.maxpool(x)
        return x, indices