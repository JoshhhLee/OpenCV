"""
The bricks you built in Lesson 01, saved as a module so later lessons can
`from blocks import ...` instead of copying them again.

If you change a brick here (e.g. a different activation), every lesson that
imports it will use your version.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Pretrained networks were trained on images normalised with these values (Lesson 02).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(image_bgr):
    """OpenCV BGR uint8 [H, W, 3]  ->  normalised float tensor [1, 3, H, W]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def count_params(module):
    """Number of trainable numbers (weights) inside a layer / model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def show(images, titles=None, cmap=None, cols=4, size=4):
    """Show one or more images inline. Accepts OpenCV BGR (3-channel) or 2-D arrays."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    titles = titles or [""] * len(images)
    rows = (len(images) + cols - 1) // cols
    cols = min(cols, len(images))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, t in zip(axes.flat, images, titles):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, matplotlib expects RGB
        ax.imshow(img, cmap=cmap or ("gray" if img.ndim == 2 else None))
        ax.set_title(t)
    plt.tight_layout()
    plt.show()


class ConvBNReLU(nn.Module):
    """Conv -> BatchNorm -> ReLU (Lesson 01, Part C)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1,
                 norm="bn", act="relu"):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=(norm is None))
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            self.norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        else:
            self.norm = nn.Identity()
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            None: nn.Identity(),
        }
        self.act = activations[act]

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """(ConvBNReLU) x 2, the UNet brick (Lesson 01, Part D)."""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(ConvBNReLU(in_ch, mid_ch), ConvBNReLU(mid_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """output = F(x) + shortcut(x), the ResNet brick (Lesson 01, Part E)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, stride=stride),
            ConvBNReLU(out_ch, out_ch, act=None),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=stride, act=None)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, the MobileNet brick (Lesson 01, Part E)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pointwise = ConvBNReLU(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
