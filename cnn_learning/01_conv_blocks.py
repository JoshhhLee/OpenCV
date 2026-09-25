"""
LESSON 01 - Convolution building blocks
=======================================
Run:  python 01_conv_blocks.py

Every CNN (ResNet, UNet, UNet++, SINet ...) is built from small LEGO bricks.
Before we build an encoder or a decoder we need to understand those bricks:

  PART A  A convolution is just an OpenCV filter (cv2.filter2D == nn.Conv2d)
  PART B  How a convolution changes the tensor SHAPE  [B, C, H, W]
  PART C  The standard brick: Conv -> BatchNorm -> ReLU      (ConvBNReLU)
  PART D  UNet's brick: two of them in a row                 (DoubleConv)
  PART E  Other bricks you will meet in papers:
            - ResidualBlock        (ResNet)
            - Dilated conv         (SINet's RF module uses these!)
            - Depthwise-separable  (MobileNet, light models)
  PART F  Look inside: feature maps of a real COD image

At the bottom there is a "TRY IT" list. Change things and re-run.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import IMAGE_SIZE, OUTPUT_DIR, find_first_image, get_device


def title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def count_params(module):
    """Number of trainable numbers (weights) inside a layer / model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# PART A - A convolution is an OpenCV filter
# ---------------------------------------------------------------------------
def part_a_conv_is_a_filter(gray):
    """
    In OpenCV you already used filters: blur, Sobel, Canny...
    A filter = a small matrix (kernel) slid over the image.

    nn.Conv2d does EXACTLY the same thing. The only difference:
      - In OpenCV YOU choose the kernel numbers (e.g. Sobel).
      - In a CNN the kernel numbers are LEARNED during training.
    """
    title("PART A - cv2.filter2D vs nn.Conv2d (same kernel -> same result)")

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    # --- OpenCV way ---
    img = gray.astype(np.float32) / 255.0
    edges_cv = cv2.filter2D(img, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_CONSTANT)

    # --- PyTorch way ---
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.from_numpy(sobel_x)  # put the Sobel numbers into the layer

    x = torch.from_numpy(img)[None, None]  # [H, W] -> [1, 1, H, W]  (batch, channel, H, W)
    with torch.no_grad():
        edges_torch = conv(x)[0, 0].numpy()

    # Note: PyTorch's "convolution" is technically cross-correlation, and so is
    # cv2.filter2D, so the kernel does not need to be flipped.
    diff = np.abs(edges_cv - edges_torch).max()
    print(f"Input shape (PyTorch) : {tuple(x.shape)}   <- [Batch, Channels, Height, Width]")
    print(f"Max difference OpenCV vs PyTorch: {diff:.6f}   (0 = identical)")

    vis = np.hstack([gray, cv2.convertScaleAbs(edges_cv, alpha=255), cv2.convertScaleAbs(edges_torch, alpha=255)])
    out = os.path.join(OUTPUT_DIR, "01_A_sobel_opencv_vs_torch.png")
    cv2.imwrite(out, vis)
    print(f"Saved (input | OpenCV | PyTorch) -> {out}")


# ---------------------------------------------------------------------------
# PART B - Shapes
# ---------------------------------------------------------------------------
def part_b_shapes():
    """
    Output size formula for Conv2d (and pooling):

        H_out = floor( (H_in + 2*padding - dilation*(kernel-1) - 1) / stride ) + 1

    The 3 cases you will use 95% of the time:
        k=3, s=1, p=1  -> SAME size         (feature extraction)
        k=3, s=2, p=1  -> HALF size         (downsample, used in encoders)
        k=1, s=1, p=0  -> SAME size         (only changes number of channels)
    """
    title("PART B - How Conv2d / pooling change the shape")

    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)  # one RGB image, 352x352
    print(f"input                          : {tuple(x.shape)}")

    layers = {
        "Conv k3 s1 p1  (3->64)        ": nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        "Conv k3 s2 p1  (3->64)        ": nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        "Conv k1 s1 p0  (3->64)        ": nn.Conv2d(3, 64, kernel_size=1),
        "Conv k3 s1 p0  (no padding!)  ": nn.Conv2d(3, 64, kernel_size=3, padding=0),
        "Conv k3 dil2 p2 (dilated)     ": nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2),
        "MaxPool k2 s2                 ": nn.MaxPool2d(kernel_size=2, stride=2),
    }
    for name, layer in layers.items():
        print(f"{name} : {tuple(layer(x).shape)}")

    print("\nKey idea: CHANNELS grow (3 -> 64 -> 128 ...) while H,W shrink.")
    print("That is exactly what an ENCODER does. A DECODER does the reverse.")


# ---------------------------------------------------------------------------
# PART C - The standard brick
# ---------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    """
    Conv -> BatchNorm -> ReLU. The most common brick in all of computer vision.

    Things YOU can modify (and papers often do):
      kernel_size : 3 is standard; 1 to mix channels; 5/7 for a bigger view
      stride      : 2 to downsample instead of max-pooling
      dilation    : >1 to see a bigger area without extra weights
      norm        : BatchNorm (default), GroupNorm (small batches), or none
      act         : ReLU (default), LeakyReLU, GELU, SiLU ...
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1,
                 norm="bn", act="relu"):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # keeps H,W the same when stride=1

        # bias=False because BatchNorm already adds its own bias
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


def part_c_conv_bn_relu():
    title("PART C - ConvBNReLU (Conv -> BatchNorm -> ReLU)")
    block = ConvBNReLU(3, 64)
    print(block)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = block(x)
    print(f"\n{tuple(x.shape)} -> {tuple(y.shape)}   params: {count_params(block):,}")
    print("Why each part:")
    print("  Conv      : finds patterns (edges, textures ...)")
    print("  BatchNorm : keeps values in a stable range -> faster, more stable training")
    print("  ReLU      : non-linearity; without it, stacking convs = just one big conv")
    print(f"After ReLU the minimum value is {y.min().item():.2f} (negatives are cut to 0)")


# ---------------------------------------------------------------------------
# PART D - UNet's brick
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    (ConvBNReLU) x 2. This is the block used at EVERY level of the original UNet,
    in both the encoder and the decoder.

        in_ch --[3x3]--> mid_ch --[3x3]--> out_ch
    """

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch),
            ConvBNReLU(mid_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


def part_d_double_conv():
    title("PART D - DoubleConv (the UNet brick)")
    block = DoubleConv(3, 64)
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"DoubleConv(3, 64): {tuple(x.shape)} -> {tuple(block(x).shape)}   params: {count_params(block):,}")
    print("Two 3x3 convs see a 5x5 area, with fewer weights than one 5x5 conv (the VGG trick).")


# ---------------------------------------------------------------------------
# PART E - Other bricks you will see in papers
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    ResNet's idea:  output = F(x) + x
    The '+ x' (the shortcut) lets gradients flow straight back, so very deep
    networks (ResNet50 = 50 layers) can still be trained.
    If the shape changes, the shortcut uses a 1x1 conv to match it.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, stride=stride),
            ConvBNReLU(out_ch, out_ch, act=None),  # no ReLU before the addition
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=stride, act=None)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class DepthwiseSeparableConv(nn.Module):
    """
    Split one 3x3 conv into two cheaper steps:
      1) depthwise : one 3x3 filter PER channel (groups=in_ch), no mixing
      2) pointwise : a 1x1 conv that mixes the channels
    Much fewer parameters. Used in MobileNet / light real-time models.
    """

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


def receptive_field(kernel_sizes, dilations=None):
    """Receptive field of a stack of stride-1 convs: how many input pixels one output pixel 'sees'."""
    dilations = dilations or [1] * len(kernel_sizes)
    rf = 1
    for k, d in zip(kernel_sizes, dilations):
        rf += d * (k - 1)
    return rf


def part_e_other_blocks():
    title("PART E - Other bricks: Residual, Dilated, Depthwise-separable")
    x = torch.randn(1, 64, 88, 88)

    blocks = {
        "ConvBNReLU 3x3 (64->128)      ": ConvBNReLU(64, 128),
        "DoubleConv       (64->128)    ": DoubleConv(64, 128),
        "ResidualBlock    (64->128)    ": ResidualBlock(64, 128),
        "ResidualBlock s2 (64->128)    ": ResidualBlock(64, 128, stride=2),
        "Dilated 3x3 d=3  (64->128)    ": ConvBNReLU(64, 128, dilation=3),
        "DepthwiseSep     (64->128)    ": DepthwiseSeparableConv(64, 128),
    }
    print(f"input: {tuple(x.shape)}")
    for name, block in blocks.items():
        print(f"{name}: out {tuple(block(x).shape)}   params {count_params(block):>8,}")

    print("\nReceptive field (how big an area one output pixel sees):")
    print(f"  3 x (3x3) normal              : {receptive_field([3, 3, 3])} px")
    print(f"  3 x (3x3) dilation 1,3,5      : {receptive_field([3, 3, 3], [1, 3, 5])} px  <- same weights, much bigger view")
    print("SINet's RF (receptive field) module runs several dilated branches in parallel")
    print("so it can see small AND large camouflaged objects. We will build it in a later lesson.")


# ---------------------------------------------------------------------------
# PART F - Look inside
# ---------------------------------------------------------------------------
def part_f_feature_maps(image_bgr):
    """
    Pass a real image through ONE random (untrained) ConvBNReLU and save 16 of
    the 32 output channels as a grid. Each channel = one 'detector'.
    Untrained filters already respond to edges/colours; training makes them useful.
    """
    title("PART F - Feature maps of a real image")
    device = get_device()
    torch.manual_seed(0)

    img = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # OpenCV: [H, W, C]  ->  PyTorch: [B, C, H, W]
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    block = ConvBNReLU(3, 32).to(device).eval()
    with torch.no_grad():
        feats = block(x)[0].cpu().numpy()  # [32, H, W]
    print(f"device: {device}   input {tuple(x.shape)} -> features {tuple(feats.shape)}")

    tiles = []
    for c in range(16):
        f = feats[c]
        f = (f - f.min()) / (f.max() - f.min() + 1e-6)
        tile = cv2.applyColorMap((f * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        tiles.append(cv2.resize(tile, (176, 176)))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])

    out = os.path.join(OUTPUT_DIR, "01_F_feature_maps.png")
    cv2.imwrite(out, grid)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_F_input.png"), img)
    print(f"Saved 16 feature maps -> {out}")


def load_sample_image():
    path = find_first_image()
    if path is not None:
        image = cv2.imread(path)
        if image is not None:
            print(f"Using dataset image: {path}")
            return image
    print("Dataset image not found -> using a synthetic image (fix DATASET_ROOT in config.py)")
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), (60, 120, 80), np.uint8)
    noise = np.random.default_rng(0).integers(0, 40, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    cv2.circle(image, (176, 176), 70, (70, 135, 90), -1)  # a "camouflaged" circle
    return image


if __name__ == "__main__":
    image = load_sample_image()
    gray = cv2.cvtColor(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)

    part_a_conv_is_a_filter(gray)
    part_b_shapes()
    part_c_conv_bn_relu()
    part_d_double_conv()
    part_e_other_blocks()
    part_f_feature_maps(image)

    title("TRY IT (change one thing, re-run, and watch the shapes / params)")
    print("""
 1. PART A: replace sobel_x with a blur kernel np.ones((3,3))/9 or its transpose (Sobel-y).
 2. PART B: add  nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            (this is the very first layer of ResNet50). What is the output size?
 3. PART C: ConvBNReLU(3, 64, act="gelu") or norm="gn". Does the param count change?
 4. PART E: which block has the fewest params for 64->128? Why?
 5. PART E: receptive_field([3,3,3,3], [1,2,4,8]). How big is it now?
 6. PART F: use ConvBNReLU(3, 32, kernel_size=7). Do the feature maps look smoother?

Next lesson (02): stack these bricks into an ENCODER and print the shape at each stage.
""")
