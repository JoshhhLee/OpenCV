"""
LESSON 00 - Check your setup
============================
Run this first:  python 00_check_setup.py

It checks:
  1. Python, PyTorch and OpenCV are installed
  2. PyTorch can see your GPU (CUDA)
  3. Your COD dataset folder exists, and what is inside it

Copy the printed output and send it back to me. The folder layout tells me
how to write the Dataset loader in a later lesson.
"""
import os
import sys

from config import DATASET_ROOT, OUTPUT_DIR


def check_libraries():
    print("=" * 60)
    print("1) LIBRARIES")
    print("=" * 60)
    print(f"Python  : {sys.version.split()[0]}")

    try:
        import torch
        print(f"PyTorch : {torch.__version__}")
    except ImportError:
        print("PyTorch : NOT INSTALLED -> see README.md, 'Setup' section")
        return False

    try:
        import cv2
        print(f"OpenCV  : {cv2.__version__}")
    except ImportError:
        print("OpenCV  : NOT INSTALLED -> pip install opencv-python")

    try:
        import numpy
        print(f"NumPy   : {numpy.__version__}")
    except ImportError:
        print("NumPy   : NOT INSTALLED -> pip install numpy")
    return True


def check_gpu():
    import torch

    print("\n" + "=" * 60)
    print("2) GPU")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("CUDA is NOT available: PyTorch will use the CPU (slow for training).")
        print("If you have an NVIDIA GPU, reinstall PyTorch with CUDA (see README.md).")
        return

    print(f"CUDA version (PyTorch build): {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}  |  memory: {props.total_memory / 1024**3:.1f} GB")

    # A tiny test: multiply two matrices on the GPU.
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = a @ b
    print(f"GPU test OK: result shape {tuple(c.shape)} on {c.device}")


def check_dataset():
    print("\n" + "=" * 60)
    print("3) DATASET")
    print("=" * 60)
    print(f"DATASET_ROOT = {DATASET_ROOT}")
    if not os.path.isdir(DATASET_ROOT):
        print("Folder NOT found. Fix DATASET_ROOT in config.py.")
        return

    # Show the folder tree (3 levels deep) with the number of files in each folder.
    base_depth = DATASET_ROOT.rstrip("\\/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(DATASET_ROOT):
        depth = dirpath.count(os.sep) - base_depth
        if depth > 3:
            dirnames[:] = []  # don't go deeper
            continue
        dirnames.sort()
        indent = "    " * depth
        name = os.path.basename(dirpath) or dirpath
        example = f"  e.g. {sorted(filenames)[0]}" if filenames else ""
        print(f"{indent}{name}/  ({len(filenames)} files){example}")


def save_sample_overlay():
    """Find one image + its mask and save them side by side, so you can see the data."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    image_path, mask_path = None, None
    for dirpath, _, filenames in os.walk(DATASET_ROOT):
        folder = dirpath.lower()
        if image_path is None and ("imgs" in folder or "image" in folder):
            jpgs = sorted(f for f in filenames if f.lower().endswith((".jpg", ".png")))
            if jpgs:
                image_path = os.path.join(dirpath, jpgs[0])
                stem = os.path.splitext(jpgs[0])[0]
                # Look for a mask with the same file name in a GT folder next to it.
                parent = os.path.dirname(dirpath)
                for sibling in sorted(os.listdir(parent)):
                    if "gt" in sibling.lower() or "mask" in sibling.lower():
                        candidate = os.path.join(parent, sibling, stem + ".png")
                        if os.path.isfile(candidate):
                            mask_path = candidate
                            break
                break

    if image_path is None:
        print("\nCould not find an image folder automatically (that's fine).")
        return

    print(f"\nSample image: {image_path}")
    print(f"Sample mask : {mask_path}")

    image = cv2.imread(image_path)
    if image is None:
        return
    panels = [image]
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Paint the camouflaged object red on top of the image.
        overlay = image.copy()
        overlay[mask > 127] = (0.5 * overlay[mask > 127] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        panels += [mask_bgr, overlay]

    out = os.path.join(OUTPUT_DIR, "00_sample.png")
    cv2.imwrite(out, np.hstack(panels))
    print(f"Saved preview (image | mask | overlay) -> {out}")


if __name__ == "__main__":
    if check_libraries():
        check_gpu()
    check_dataset()
    save_sample_overlay()
