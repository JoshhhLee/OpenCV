"""
COD10K dataset + data augmentation with OpenCV (Lesson 04).

Why is this a .py file and not a notebook cell?
On Windows, a DataLoader with num_workers > 0 starts new Python processes, and
they can only load classes that live in an importable file, not in a notebook.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from blocks import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Augmentations: each one changes the image AND the mask in exactly the same way
# (except colour changes, which only touch the image).
# ---------------------------------------------------------------------------
def random_flip(image, mask, rng, p=0.5):
    """Mirror left <-> right."""
    if rng.random() < p:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    return image, mask


def random_scale_crop(image, mask, rng, min_area=0.6):
    """Cut out a random box covering min_area..100% of the image (a zoom-in)."""
    h, w = mask.shape
    area = rng.uniform(min_area, 1.0)
    ch, cw = int(h * np.sqrt(area)), int(w * np.sqrt(area))
    y0 = rng.integers(0, h - ch + 1)
    x0 = rng.integers(0, w - cw + 1)
    return image[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]


def random_rotate(image, mask, rng, max_deg=15, p=0.5):
    """Rotate around the centre by -max_deg..+max_deg degrees."""
    if rng.random() >= p:
        return image, mask
    h, w = mask.shape
    angle = rng.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def color_jitter(image, rng, brightness=0.2, contrast=0.2, saturation=0.2):
    """Random brightness / contrast / saturation (image only, the mask does not change)."""
    img = image.astype(np.float32)
    img = img * rng.uniform(1 - contrast, 1 + contrast) + (rng.uniform(-brightness, brightness) * 255)
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - saturation, 1 + saturation), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment(image, mask, rng):
    """The full training augmentation. Edit this to try other combinations."""
    image, mask = random_flip(image, mask, rng)
    image, mask = random_scale_crop(image, mask, rng)
    image, mask = random_rotate(image, mask, rng)
    image = color_jitter(image, rng)
    return image, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CODDataset(Dataset):
    """
    pairs : list of (image_path, mask_path), e.g. from config.cod10k_pairs()
    size  : images and masks are resized to size x size
    train : True -> random augmentation every time an image is loaded

    Returns (image [3, size, size] normalised float, mask [1, size, size] with values 0 or 1).
    """

    def __init__(self, pairs, size=352, train=False):
        self.pairs = list(pairs)
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def load(self, index):
        """Read one (image BGR, mask 0..255) pair from disk, without any processing."""
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not read {image_path} or {mask_path}")
        return image, mask

    def __getitem__(self, index):
        image, mask = self.load(index)
        if self.train:
            image, mask = augment(image, mask, np.random.default_rng())  # new randomness every call

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(rgb).permute(2, 0, 1)                      # [3, H, W]
        mask_t = torch.from_numpy((mask > 127).astype(np.float32))[None]       # [1, H, W]
        return image_t, mask_t
