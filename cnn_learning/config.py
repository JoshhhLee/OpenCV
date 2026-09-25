"""
Shared settings for every lesson in cnn_learning/.
Change the path here once, and every lesson picks it up.
"""
import os

import torch

# Where your COD dataset lives on your PC.
# The r"..." (raw string) stops Windows backslashes being read as escape codes.
# (You can also set the COD_DATASET_ROOT environment variable instead of editing this line.)
DATASET_ROOT = os.environ.get("COD_DATASET_ROOT", r"D:\PhD\Dataset\COD1K Dataset")

# Your layout (from 00_check_setup):
#   COD10K-v3/
#     Train/  Image/ (6000 .jpg)  GT_Object/  GT_Edge/  GT_Instance/  (.png)
#     Test/   Image/ (4000 .jpg)  GT_Object/  GT_Edge/  GT_Instance/  (.png)
COD10K_ROOT = os.path.join(DATASET_ROOT, "COD10K-v3")

# Most COD papers (SINet, SINet-V2, PFNet, ...) resize inputs to 352x352.
IMAGE_SIZE = 352

# Folder where lessons save pictures / results.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    """Use the GPU if PyTorch can see one, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cod10k_pairs(split="Train", mask_folder="GT_Object", cam_only=True):
    """
    Return a sorted list of (image_path, mask_path) for COD10K.

    split       : "Train" or "Test"
    mask_folder : "GT_Object" (the object mask, what we predict) or "GT_Edge" (its outline)
    cam_only    : COD10K also has NON-camouflaged images ("-NonCAM-" in the file name).
                  Their masks are empty. The standard COD protocol (SINet) keeps only the
                  camouflaged ones: 3040 train and 2026 test images.
    """
    image_dir = os.path.join(COD10K_ROOT, split, "Image")
    mask_dir = os.path.join(COD10K_ROOT, split, mask_folder)
    if not os.path.isdir(image_dir):
        return []

    pairs = []
    for name in sorted(os.listdir(image_dir)):
        if not name.lower().endswith((".jpg", ".png")):
            continue
        if cam_only and "-CAM-" not in name:
            continue
        mask_path = os.path.join(mask_dir, os.path.splitext(name)[0] + ".png")
        if os.path.isfile(mask_path):
            pairs.append((os.path.join(image_dir, name), mask_path))
    return pairs


def find_first_image(root=DATASET_ROOT, exts=(".jpg", ".jpeg", ".png")):
    """Return the path of one sample image (a COD10K test image if possible), or None."""
    pairs = cod10k_pairs("Test")
    if pairs:
        return pairs[0][0]
    if not os.path.isdir(root):
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(exts):
                return os.path.join(dirpath, name)
    return None
