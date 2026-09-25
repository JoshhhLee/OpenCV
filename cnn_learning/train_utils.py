"""
Loss, metrics and the training loop from Lesson 04, saved as a module so later
lessons can `from train_utils import ...`.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def bce_iou_loss(logits, target):
    """
    BCE  : judges every pixel on its own ("is this pixel object?")
    IoU  : judges the whole shape ("how much do prediction and GT overlap?")
    Together they work better than either alone, especially for small objects,
    where BCE is dominated by the (huge) background.
    """
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    union = (prob + target).sum(dim=(2, 3)) - inter
    iou_loss = 1 - (inter + 1) / (union + 1)      # +1 avoids 0/0 on empty masks
    return bce + iou_loss.mean()


@torch.no_grad()
def batch_metrics(logits, target, threshold=0.5):
    """
    Per-image metrics for a batch. Returns three lists (one value per image).
      MAE  : mean |prob - GT|, the standard COD metric (lower is better)
      IoU  : overlap of the thresholded mask with the GT (higher is better)
      Dice : 2*overlap / (pred + GT), also called F1 (higher is better)
    """
    prob = torch.sigmoid(logits.float())
    pred = (prob > threshold).float()
    mae = (prob - target).abs().mean(dim=(1, 2, 3))
    inter = (pred * target).sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    iou = torch.where(total - inter > 0, inter / (total - inter).clamp(min=1), torch.ones_like(inter))
    dice = torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(inter))
    return mae.tolist(), iou.tolist(), dice.tolist()


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn=None):
    """One pass over the training data. Returns the average loss."""
    loss_fn = loss_fn or bce_iou_loss
    use_amp = device.type == "cuda"
    model.train()
    total, count = 0.0, 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, per_image=False):
    """Average MAE / IoU / Dice over a dataset (no augmentation, no gradients)."""
    use_amp = device.type == "cuda"
    model.eval()
    maes, ious, dices = [], [], []
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        m, i, d = batch_metrics(logits, masks)
        maes += m
        ious += i
        dices += d
    result = {"mae": float(np.mean(maes)), "iou": float(np.mean(ious)), "dice": float(np.mean(dices))}
    if per_image:
        result["per_image_iou"] = ious
    return result
