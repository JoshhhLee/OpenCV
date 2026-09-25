"""
Shared settings for every lesson in cnn_learning/.
Change the path here once, and every lesson file picks it up.
"""
import os

import torch

# Where your COD dataset lives on your PC.
# The r"..." (raw string) stops Windows backslashes being read as escape codes.
DATASET_ROOT = r"D:\PhD\Dataset\COD1K Dataset"

# Most COD papers (SINet, SINet-V2, PFNet, ...) resize inputs to 352x352.
IMAGE_SIZE = 352

# Folder where lessons save pictures / results.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    """Use the GPU if PyTorch can see one, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_first_image(root=DATASET_ROOT, exts=(".jpg", ".jpeg", ".png")):
    """Return the path of the first image under `root`, or None if there isn't one."""
    if not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(exts):
                return os.path.join(dirpath, name)
    return None
