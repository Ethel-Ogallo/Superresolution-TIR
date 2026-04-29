import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxEncoder(nn.Module):
    """
    Encoder for continuous + categorical geophysical priors.
    """

    def __init__(self, in_channels, base_channels=64):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

    def forward(self, aux):
        x = self.stem(aux)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return x