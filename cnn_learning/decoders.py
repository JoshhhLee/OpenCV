"""
The decoder you built in Lesson 03, saved as a module so later lessons can
`from decoders import ...`.

Decoder contract:
    decoder([f1, f2, f3, f4, f5], out_size)  ->  mask logits [B, 1, H, W]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import ConvBNReLU, DoubleConv

def make_up(kind, ch):
    """The upsampling step: H, W -> 2H, 2W."""
    if kind == "transpose":
        return nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)  # learnable
    if kind == "nearest":
        return nn.Upsample(scale_factor=2, mode="nearest")
    if kind == "bilinear":
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    raise ValueError(kind)


class DecoderBlock(nn.Module):
    """
    One decoder step:  upsample  ->  merge the skip feature  ->  conv.

    in_ch   : channels coming up from the deeper decoder step
    skip_ch : channels of the encoder feature at this scale (0 = no skip)
    out_ch  : channels this block outputs
    up      : "bilinear" | "nearest" | "transpose"
    merge   : "concat" (UNet)  |  "add" (FPN / many COD models)
    """

    def __init__(self, in_ch, skip_ch, out_ch, up="bilinear", merge="concat"):
        super().__init__()
        self.up = make_up(up, in_ch)
        self.merge = merge if skip_ch > 0 else None
        if self.merge == "concat":
            self.fuse = DoubleConv(in_ch + skip_ch, out_ch)      # channels stack up
        elif self.merge == "add":
            self.skip_proj = ConvBNReLU(skip_ch, in_ch, kernel_size=1)  # match channels first
            self.fuse = DoubleConv(in_ch, out_ch)
        else:
            self.fuse = DoubleConv(in_ch, out_ch)                # no skip at all

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.merge is None or skip is None:
            return self.fuse(x)
        if x.shape[-2:] != skip.shape[-2:]:  # e.g. odd input sizes
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.merge == "concat":
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + self.skip_proj(skip)
        return self.fuse(x)


class UNetDecoder(nn.Module):
    """
    Walks back up the encoder features, deepest first:

        f5 -> block(+f4) -> block(+f3) -> block(+f2) -> block(+f1) -> head -> mask logits

    encoder_channels : the encoder's `.channels`, e.g. [64, 128, 256, 512, 1024]
    decoder_channels : output channels of each block, deep -> shallow (default: mirror the encoder)
    use_skips        : one True/False per block, deep -> shallow
    """

    def __init__(self, encoder_channels, decoder_channels=None, up="bilinear", merge="concat",
                 use_skips=None, num_classes=1):
        super().__init__()
        skip_channels = list(encoder_channels[:-1])[::-1]        # [f4, f3, f2, f1] channels
        decoder_channels = list(decoder_channels or skip_channels)
        use_skips = list(use_skips or [True] * len(skip_channels))
        self.use_skips = use_skips

        blocks = []
        in_ch = encoder_channels[-1]                              # start from f5
        for skip_ch, out_ch, use in zip(skip_channels, decoder_channels, use_skips):
            blocks.append(DecoderBlock(in_ch, skip_ch if use else 0, out_ch, up=up, merge=merge))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, num_classes, kernel_size=1)  # 1 channel = "object" score

    def forward(self, feats, out_size=None):
        x = feats[-1]
        skips = feats[-2::-1]                                     # [f4, f3, f2, f1]
        for block, skip, use in zip(self.blocks, skips, self.use_skips):
            x = block(x, skip if use else None)
        x = self.head(x)
        if out_size is not None and x.shape[-2:] != tuple(out_size):
            # ResNet's f1 is stride 2, so the last step still needs a 2x upsample
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x  # logits: apply torch.sigmoid() to get a 0..1 mask


class SegModel(nn.Module):
    """Any encoder + any decoder = a segmentation model."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x), out_size=x.shape[-2:])
