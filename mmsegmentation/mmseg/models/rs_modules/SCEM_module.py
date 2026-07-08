from mmengine.model import BaseModule
from mmseg.registry import MODELS

@MODELS.register_module()
class SCEM(BaseModule):
    def __init__(self, dim):
        super().__init__()

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

        x_h = F.adaptive_avg_pool2d(x, (H, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, W))

        x_h = F.interpolate(self.conv1(x_h), (H, W))
        x_w = F.interpolate(self.conv2(x_w), (H, W))

        context = x_h + x_w
        gate = self.gate(context)

        out = x * gate
        out = self.fuse(out)

        return out + identity