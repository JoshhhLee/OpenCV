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


class UNetPlusPlusDecoder(nn.Module):
    """
    UNet++ (Zhou et al., 2018): nested, dense skip connections.

    Nodes X[i][j]:  i = level (0 = highest resolution), j = column.
      X[i][0]  = encoder feature f(i+1)
      X[i][j]  = conv( concat( X[i][0], ..., X[i][j-1],  up(X[i+1][j-1]) ) )
    The final mask comes from X[0][L]. With deep supervision, every X[0][j] gets its own head.

    encoder_channels : the encoder's `.channels` (5 values)
    level_channels   : channels of the new nodes at levels 0..3 (like decoder_channels in UNet)
    deep_supervision : train with an output head on every X[0][j]
    """

    def __init__(self, encoder_channels, level_channels=(32, 64, 128, 256), deep_supervision=False, num_classes=1):
        super().__init__()
        e = list(encoder_channels)
        c = list(level_channels)
        self.L = len(e) - 1                    # 4 decoder columns for a 5-level encoder
        self.deep_supervision = deep_supervision

        self.nodes = nn.ModuleDict()
        for j in range(1, self.L + 1):
            for i in range(0, self.L + 1 - j):
                below = e[i + 1] if j == 1 else c[i + 1]          # channels of X[i+1][j-1]
                in_ch = e[i] + c[i] * (j - 1) + below             # X[i][0] + X[i][1..j-1] + up(below)
                self.nodes[f"x{i}_{j}"] = DoubleConv(in_ch, c[i])

        n_heads = self.L if deep_supervision else 1
        self.heads = nn.ModuleList([nn.Conv2d(c[0], num_classes, kernel_size=1) for _ in range(n_heads)])

    def forward(self, feats, out_size=None, all_heads=False):
        X = {(i, 0): f for i, f in enumerate(feats)}
        for j in range(1, self.L + 1):
            for i in range(0, self.L + 1 - j):
                target = X[(i, 0)].shape[-2:]
                up = F.interpolate(X[(i + 1, j - 1)], size=target, mode="bilinear", align_corners=False)
                inputs = [X[(i, k)] for k in range(j)] + [up]
                X[(i, j)] = self.nodes[f"x{i}_{j}"](torch.cat(inputs, dim=1))

        def finish(logits):
            if out_size is not None and logits.shape[-2:] != tuple(out_size):
                logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
            return logits

        if self.deep_supervision and (self.training or all_heads):
            return [finish(head(X[(0, j)])) for j, head in zip(range(1, self.L + 1), self.heads)]
        return finish(self.heads[-1](X[(0, self.L)]))
