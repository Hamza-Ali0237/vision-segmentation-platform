import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)