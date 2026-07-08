from mmengine.model import BaseModule
from mmseg.registry import MODELS

@MODELS.register_module()
class BARM(BaseModule):
    def __init__(self, dim):
        super().__init__()

        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, 1)

        self.edge = SobelEdge(dim)

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
