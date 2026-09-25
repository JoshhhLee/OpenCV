"""
The encoders you built in Lesson 02, saved as a module so later lessons can
`from encoders import ...`.

Every encoder follows the same contract:
    encoder(image [B, 3, H, W])  ->  [f1, f2, f3, f4, f5]   (big/shallow -> small/deep)
and has a `.channels` list with the number of channels of each feature.
"""
import torch.nn as nn

from blocks import ConvBNReLU, DepthwiseSeparableConv, DoubleConv, ResidualBlock


class UNetEncoder(nn.Module):
    """The original UNet encoder (Lesson 02, Part A). Strides 1, 2, 4, 8, 16."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.stage1 = DoubleConv(in_ch, 64)
        self.stage2 = DoubleConv(64, 128)
        self.stage3 = DoubleConv(128, 256)
        self.stage4 = DoubleConv(256, 512)
        self.stage5 = DoubleConv(512, 1024)
        self.channels = [64, 128, 256, 512, 1024]

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(self.pool(f1))
        f3 = self.stage3(self.pool(f2))
        f4 = self.stage4(self.pool(f3))
        f5 = self.stage5(self.pool(f4))
        return [f1, f2, f3, f4, f5]


BLOCKS = {
    "double": DoubleConv,
    "residual": ResidualBlock,
    "dws": DepthwiseSeparableConv,
}


def make_down(kind, ch):
    if kind == "maxpool":
        return nn.MaxPool2d(2)
    if kind == "avgpool":
        return nn.AvgPool2d(2)
    if kind == "conv":
        return ConvBNReLU(ch, ch, kernel_size=3, stride=2)
    raise ValueError(kind)


class ConfigurableEncoder(nn.Module):
    """Encoder with the knobs from Lesson 02, Part B: channels, block, down."""

    def __init__(self, in_ch=3, channels=(64, 128, 256, 512, 1024), block="double", down="maxpool"):
        super().__init__()
        Block = BLOCKS[block]
        self.channels = list(channels)
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = in_ch
        for i, ch in enumerate(channels):
            self.downs.append(nn.Identity() if i == 0 else make_down(down, prev))
            self.stages.append(Block(prev, ch))
            prev = ch

    def forward(self, x):
        feats = []
        for down, stage in zip(self.downs, self.stages):
            x = stage(down(x))
            feats.append(x)
        return feats


class ResNet50Encoder(nn.Module):
    """torchvision ResNet50 without its classifier (Lesson 02, Part C). Strides 2, 4, 8, 16, 32."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        net = None
        if pretrained:
            try:
                net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except Exception as e:  # e.g. no internet
                print(f"Could not download pretrained weights ({type(e).__name__}) -> using random weights")
        if net is None:
            net = resnet50(weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.layer1(self.pool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return [f1, f2, f3, f4, f5]
