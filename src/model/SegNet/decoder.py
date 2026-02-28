import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(Decoder, self).__init__()

        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.dec_block = nn.ModuleList()

        current_in_channels = in_channels

        for i in range(num_conv_layers):

            if i+1 == num_conv_layers:
                current_out_channels = out_channels
            else:
                current_out_channels = in_channels
            
            layer = nn.Sequential(
                nn.Conv2d(current_in_channels, current_out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(current_out_channels),
                nn.ReLU(inplace=True)
            )

            self.dec_block.append(layer)

            current_in_channels = current_out_channels

    
    def forward(self, x, indices):
        x = self.max_unpool(x, indices)

        for layer in self.dec_block:
            x = layer(x)

        return x