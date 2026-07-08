import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdge(nn.Module):
    def __init__(self, channels):
        super().__init__()

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]
        ], dtype=torch.float32)

        self.conv_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        self.conv_x.weight.data = sobel_x.view(1,1,3,3).repeat(channels,1,1,1)
        self.conv_y.weight.data = sobel_y.view(1,1,3,3).repeat(channels,1,1,1)

        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class BARM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 1. lightweight texture extractor
        self.dw_conv = nn.Conv2d(
            dim, dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False
        )

        self.pw_conv = nn.Conv2d(dim, dim, 1)

        # 2. edge extractor
        self.edge = SobelEdge(dim)

        # 3. boundary attention (very lightweight)
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x

        x = self.dw_conv(x)
        x = self.pw_conv(x)

        edge = self.edge(x)

        attn = self.attn(edge)

        x = x * attn

        x = self.fuse(x)

        return x + identity
