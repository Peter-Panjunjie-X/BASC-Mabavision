import torch
import torch.nn as nn
import torch.nn.functional as F

class SCEM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # horizontal context
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        # vertical context
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x

        B, C, H, W = x.shape

        # horizontal
        x_h = self.pool_h(x)
        x_h = self.conv1(x_h)
        x_h = F.interpolate(x_h, (H, W))

        # vertical
        x_w = self.pool_w(x)
        x_w = self.conv2(x_w)
        x_w = F.interpolate(x_w, (H, W))

        context = x_h + x_w

        gate = self.gate(context)

        out = x * gate

        out = self.fuse(out)

        return out + identity