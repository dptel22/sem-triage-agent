"""Train and Evaluate U-Net Baseline for Carinthia-S SEM Defect Segmentation.

Trains a lightweight ResNet18-UNet baseline on nominal_train + class6_train (3,618 images)
using BCE + Dice loss at 480x480 resolution, zeroing the 3px right border artifact (x >= 477).
Runs on local NVIDIA CUDA GPU (RTX 4050 Laptop GPU).

Key Gate 1 Requirements Addressed:
1. Robust Connected Component BBoxes:
   - Prevents corner-bug stretching by isolating connected components.
   - Filters components below min_area (default: 16 pixels).
   - Exports:
     - largest_component_bbox: [x1, y1, x2, y2] of largest defect component.
     - filtered_foreground_bbox: tight bbox of all components >= min_area (dust-free).
     - overall_bbox: raw all-foreground bbox (retained for baseline comparison).
     - component_boxes: all components >= min_area with individual bounding boxes and pixel areas.
2. Complete Split Coverage:
   - All 20 Gate 0 images.
   - All 430 nominal_val images.
   - All 430 nominal_test images.
   - All 67 rare_holdout images (Classes 1, 2, 5) with seen_in_gate0 flag.
3. Dual Threshold Reporting:
   - Primary: mask >= 128 with x >= 477 zeroed out.
   - Secondary: mask > 0 with x >= 477 zeroed out.
4. Reproducibility:
   - Deterministic seed 42 across PyTorch, CUDA, CuDNN, NumPy, and Python.
   - Full environment and library version manifest recorded in output JSON.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_MANIFEST_PATH = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
GATE0_MANIFEST_PATH = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "gate1"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "gate1"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


def set_seed(seed: int = 42):
    """Sets deterministic random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_info() -> Dict[str, Any]:
    """Records exact library versions and hardware configuration."""
    import torchvision
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "fixed_seed": 42,
    }


# ---------------------------------------------------------------------------
# Top-Level PyTorch Dataset & Model Definitions (Required for Windows pickling)
# ---------------------------------------------------------------------------
class SEMDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], root: Path):
        self.samples = samples
        self.root = root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        sid = item["id"]
        lbl = item["class_label"]

        img_p = self.root / "images" / f"{sid}.jpg"
        mask_p = self.root / "masks" / f"{sid}.png"

        img = Image.open(img_p).convert("L")
        img_arr = np.array(img, dtype=np.float32) / 255.0  # (480, 480)

        mask = Image.open(mask_p).convert("L")
        mask_arr = np.array(mask, dtype=np.uint8)

        # Primary GT: mask >= 128, x >= 477 zeroed out
        gt_primary = (mask_arr >= 128).astype(np.float32)
        gt_primary[:, 477:] = 0.0

        # Secondary GT: mask > 0, x >= 477 zeroed out
        gt_secondary = (mask_arr > 0).astype(np.float32)
        gt_secondary[:, 477:] = 0.0

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
        gt_pri_tensor = torch.from_numpy(gt_primary).unsqueeze(0)
        gt_sec_tensor = torch.from_numpy(gt_secondary).unsqueeze(0)

        return {
            "id": sid,
            "label": lbl,
            "image": img_tensor,
            "mask_primary": gt_pri_tensor,
            "mask_secondary": gt_sec_tensor,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet18UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapt first conv for 1-channel grayscale SEM
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.init_conv.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.enc1 = resnet.layer1  # 64 ch, /4
        self.enc2 = resnet.layer2  # 128 ch, /8
        self.enc3 = resnet.layer3  # 256 ch, /16
        self.enc4 = resnet.layer4  # 512 ch, /32

        # Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.relu(self.bn1(self.init_conv(x)))  # (B, 64, 240, 240)
        p0 = self.maxpool(e0)                         # (B, 64, 120, 120)
        e1 = self.enc1(p0)                            # (B, 64, 120, 120)
        e2 = self.enc2(e1)                            # (B, 128, 60, 60)
        e3 = self.enc3(e2)                            # (B, 256, 30, 30)
        e4 = self.enc4(e3)                            # (B, 512, 15, 15)

        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))  # (B, 256, 30, 30)
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))  # (B, 128, 60, 60)
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))  # (B, 64, 120, 120)
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))  # (B, 64, 240, 240)
        d0 = self.dec0(self.up0(d1))                           # (B, 32, 480, 480)
        logits = self.head(d0)                                # (B, 1, 480, 480)
        return logits


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()
    return bce + dice_loss


def extract_component_bboxes(
    binary_mask: np.ndarray,
    min_area: int = 16,
) -> Dict[str, Any]:
    """Extracts connected component bounding boxes, largest component, and filtered foreground bbox.

    Addresses the 'corner bug' / single-pixel border artifact from Gate 0:
    - Eliminates isolated pixels (< min_area) that stretched the oracle bounding boxes.
    - Computes largest_component_bbox and filtered_foreground_bbox.
    - All bboxes follow inclusive [x1, y1, x2, y2] convention.
    """
    total_fg_pixels = int(binary_mask.sum())
    if total_fg_pixels == 0:
        return {
            "overall_bbox": None,
            "largest_component_bbox": None,
            "filtered_foreground_bbox": None,
            "component_boxes": [],
            "filtered_pixel_count": 0,
        }

    # Raw overall bbox (inclusive)
    y_idx, x_idx = np.where(binary_mask > 0)
    overall_bbox = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())]

    # Connected component analysis (8-connectivity)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    component_boxes = []
    largest_box = None
    largest_area = 0

    valid_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for lbl_idx in range(1, num_labels):
        area = int(stats[lbl_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            x1 = int(stats[lbl_idx, cv2.CC_STAT_LEFT])
            y1 = int(stats[lbl_idx, cv2.CC_STAT_TOP])
            x2 = int(x1 + stats[lbl_idx, cv2.CC_STAT_WIDTH] - 1)
            y2 = int(y1 + stats[lbl_idx, cv2.CC_STAT_HEIGHT] - 1)
            bbox = [x1, y1, x2, y2]
            component_boxes.append({"bbox": bbox, "area": area})
            valid_mask[labels == lbl_idx] = 1

            if area > largest_area:
                largest_area = area
                largest_box = bbox

    # Sort components by area descending
    component_boxes = sorted(component_boxes, key=lambda c: c["area"], reverse=True)

    # Filtered foreground bbox enclosing all valid components
    filtered_pixel_count = int(valid_mask.sum())
    filtered_bbox = None
    if filtered_pixel_count > 0:
        vy_idx, vx_idx = np.where(valid_mask > 0)
        filtered_bbox = [int(vx_idx.min()), int(vy_idx.min()), int(vx_idx.max()), int(vy_idx.max())]
    elif largest_box is not None:
        filtered_bbox = largest_box

    return {
        "overall_bbox": overall_bbox,
        "largest_component_bbox": largest_box,
        "filtered_foreground_bbox": filtered_bbox,
        "component_boxes": component_boxes,
        "filtered_pixel_count": filtered_pixel_count,
    }


def run_training_and_evaluation(
    data_root_str: str,
    splits_dict: Dict[str, List[Dict[str, Any]]],
    gate0_ids: List[str],
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    min_component_area: int = 16,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """Runs complete U-Net baseline training, validation checkpointing, and multi-split evaluation."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Compute device: {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU Hardware: {torch.cuda.get_device_name(0)}")

    data_root = Path(data_root_str)

    # Prepare Datasets
    train_samples = splits_dict["nominal_train"] + splits_dict["class6_train"]
    val_samples = splits_dict["nominal_val"]
    print(f"[Data] Training set size: {len(train_samples)} (nominal_train: {len(splits_dict['nominal_train'])}, class6_train: {len(splits_dict['class6_train'])})")
    print(f"[Data] Validation set size: {len(val_samples)} (nominal_val: {len(splits_dict['nominal_val'])})")

    train_ds = SEMDataset(train_samples, data_root)
    val_ds = SEMDataset(val_samples, data_root)

    # Windows-friendly multiprocessing: num_workers=0 avoids spawn overhead on local NVMe
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Model, Optimizer, Scaler
    model = ResNet18UNet(in_channels=1, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_dice = -1.0
    best_weights_bytes: Optional[bytes] = None

    print(f"[Train] Training ResNet18-UNet for {epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask_primary"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(imgs)
                logits[:, :, :, 477:] = -100.0  # Zero right border
                loss = bce_dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Evaluate on nominal_val
        model.eval()
        val_dices = []
        val_ious = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask_primary"].cpu().numpy()[:, 0]

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    logits[:, :, :, 477:] = -100.0
                    probs = torch.sigmoid(logits)[:, 0].cpu().numpy()

                preds = (probs >= 0.5).astype(np.uint8)
                preds[:, :, 477:] = 0

                for i in range(len(masks)):
                    p = preds[i]
                    gt = (masks[i] >= 0.5).astype(np.uint8)
                    inter = np.logical_and(p, gt).sum()
                    union = np.logical_or(p, gt).sum()
                    iou = (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)
                    dice = (2.0 * inter / (p.sum() + gt.sum())) if (p.sum() + gt.sum()) > 0 else (1.0 if inter == 0 else 0.0)
                    val_ious.append(float(iou))
                    val_dices.append(float(dice))

        mean_dice = float(np.mean(val_dices))
        median_iou = float(np.median(val_ious))
        mean_iou = float(np.mean(val_ious))

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | nominal_val Mean Dice: {mean_dice:.4f} | Median IoU: {median_iou:.4f} | Mean IoU: {mean_iou:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            best_weights_bytes = buf.getvalue()

    total_train_sec = time.time() - start_time
    print(f"[Train] Training complete in {total_train_sec:.1f}s. Best nominal_val Mean Dice: {best_val_dice:.4f}")

    # Load best checkpoint for multi-split evaluation
    assert best_weights_bytes is not None
    buf = io.BytesIO(best_weights_bytes)
    model.load_state_dict(torch.load(buf, map_location=device))
    model.eval()

    # Evaluation Helper
    def evaluate_split(samples: List[Dict[str, Any]], split_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ds = SEMDataset(samples, data_root)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        records_out = []
        ious_primary = []
        dices_primary = []
        ious_secondary = []
        dices_secondary = []
        fp_pixels_empty = []
        empty_count = 0
        empty_zero_pred_count = 0

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device, non_blocking=True)
                m_pri = batch["mask_primary"].cpu().numpy()[:, 0]
                m_sec = batch["mask_secondary"].cpu().numpy()[:, 0]
                labels = batch["label"].tolist()
                sids = batch["id"]

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    logits[:, :, :, 477:] = -100.0
                    probs = torch.sigmoid(logits)[:, 0].cpu().numpy()

                preds = (probs >= 0.5).astype(np.uint8)
                preds[:, :, 477:] = 0

                for i in range(len(sids)):
                    sid = sids[i]
                    lbl = labels[i]
                    p = preds[i]
                    prob_map = probs[i]
                    gt_p = (m_pri[i] >= 0.5).astype(np.uint8)
                    gt_s = (m_sec[i] >= 0.5).astype(np.uint8)

                    fg_pixels = int(p.sum())
                    max_prob = float(prob_map.max())

                    # Primary metrics (mask >= 128)
                    inter_p = np.logical_and(p, gt_p).sum()
                    union_p = np.logical_or(p, gt_p).sum()
                    gt_p_sum = int(gt_p.sum())

                    if gt_p_sum == 0:
                        empty_count += 1
                        fp_pixels_empty.append(fg_pixels)
                        if fg_pixels == 0:
                            empty_zero_pred_count += 1
                        iou_p = 1.0 if fg_pixels == 0 else 0.0
                        dice_p = 1.0 if fg_pixels == 0 else 0.0
                    else:
                        iou_p = float(inter_p / union_p) if union_p > 0 else 0.0
                        dice_p = float(2.0 * inter_p / (fg_pixels + gt_p_sum)) if (fg_pixels + gt_p_sum) > 0 else 0.0

                    # Secondary metrics (mask > 0)
                    inter_s = np.logical_and(p, gt_s).sum()
                    union_s = np.logical_or(p, gt_s).sum()
                    gt_s_sum = int(gt_s.sum())
                    if gt_s_sum == 0:
                        iou_s = 1.0 if fg_pixels == 0 else 0.0
                        dice_s = 1.0 if fg_pixels == 0 else 0.0
                    else:
                        iou_s = float(inter_s / union_s) if union_s > 0 else 0.0
                        dice_s = float(2.0 * inter_s / (fg_pixels + gt_s_sum)) if (fg_pixels + gt_s_sum) > 0 else 0.0

                    ious_primary.append(iou_p)
                    dices_primary.append(dice_p)
                    ious_secondary.append(iou_s)
                    dices_secondary.append(dice_s)

                    # Extract Component Bounding Boxes
                    bbox_data = extract_component_bboxes(p, min_area=min_component_area)

                    records_out.append({
                        "id": sid,
                        "class_label": lbl,
                        "split": split_name,
                        "iou_primary": round(iou_p, 4),
                        "dice_primary": round(dice_p, 4),
                        "iou_secondary": round(iou_s, 4),
                        "dice_secondary": round(dice_s, 4),
                        "predicted_pixels": fg_pixels,
                        "filtered_pixels": bbox_data["filtered_pixel_count"],
                        "ground_truth_primary_pixels": gt_p_sum,
                        "max_prob": round(max_prob, 4),
                        "overall_bbox": bbox_data["overall_bbox"],
                        "largest_component_bbox": bbox_data["largest_component_bbox"],
                        "filtered_foreground_bbox": bbox_data["filtered_foreground_bbox"],
                        "component_boxes": bbox_data["component_boxes"],
                    })

        summary: Dict[str, Any] = {
            "total_images": len(samples),
            "median_iou_primary": round(float(np.median(ious_primary)), 4) if ious_primary else 0.0,
            "mean_iou_primary": round(float(np.mean(ious_primary)), 4) if ious_primary else 0.0,
            "mean_dice_primary": round(float(np.mean(dices_primary)), 4) if dices_primary else 0.0,
            "median_iou_secondary": round(float(np.median(ious_secondary)), 4) if ious_secondary else 0.0,
            "mean_iou_secondary": round(float(np.mean(ious_secondary)), 4) if ious_secondary else 0.0,
            "mean_dice_secondary": round(float(np.mean(dices_secondary)), 4) if dices_secondary else 0.0,
        }

        # Breakdown per defect class
        classes_present = sorted(list({s["class_label"] for s in samples}))
        class_metrics = {}
        for c in classes_present:
            c_recs = [r for r in records_out if r["class_label"] == c]
            c_ious_p = [r["iou_primary"] for r in c_recs]
            c_dices_p = [r["dice_primary"] for r in c_recs]
            c_ious_s = [r["iou_secondary"] for r in c_recs]
            c_dices_s = [r["dice_secondary"] for r in c_recs]
            class_metrics[f"class_{c}"] = {
                "count": len(c_recs),
                "median_iou_primary": round(float(np.median(c_ious_p)), 4),
                "mean_iou_primary": round(float(np.mean(c_ious_p)), 4),
                "mean_dice_primary": round(float(np.mean(c_dices_p)), 4),
                "median_iou_secondary": round(float(np.median(c_ious_s)), 4),
                "mean_iou_secondary": round(float(np.mean(c_ious_s)), 4),
                "mean_dice_secondary": round(float(np.mean(c_dices_s)), 4),
            }
        summary["per_class"] = class_metrics

        if empty_count > 0:
            summary["empty_scan_metrics"] = {
                "empty_scans_total": empty_count,
                "empty_scans_zero_predicted": empty_zero_pred_count,
                "empty_scan_specificity": round(100.0 * empty_zero_pred_count / empty_count, 2),
                "mean_false_positive_pixels": round(float(np.mean(fp_pixels_empty)), 2),
                "median_false_positive_pixels": round(float(np.median(fp_pixels_empty)), 2),
                "max_false_positive_pixels": int(np.max(fp_pixels_empty)),
            }

        return summary, records_out

    # 7. Evaluate on all splits
    print("\n[Eval] Running inference on nominal_val (430 images)...")
    val_summary, val_records = evaluate_split(splits_dict["nominal_val"], "nominal_val")

    print("[Eval] Running inference on nominal_test (430 images)...")
    test_summary, test_records = evaluate_split(splits_dict["nominal_test"], "nominal_test")

    print("[Eval] Running inference on class6_val (23 images)...")
    c6_val_summary, c6_val_records = evaluate_split(splits_dict["class6_val"], "class6_val")

    print("[Eval] Running inference on class6_test (23 images)...")
    c6_test_summary, c6_test_records = evaluate_split(splits_dict["class6_test"], "class6_test")

    print("[Eval] Running inference on rare_holdout (all 67 images)...")
    rare_summary, rare_records = evaluate_split(splits_dict["rare_holdout"], "rare_holdout")

    gate0_set = set(gate0_ids)
    for r in rare_records:
        r["seen_in_gate0"] = r["id"] in gate0_set

    # 8. Results Structure
    results = {
        "metadata": {
            "model_architecture": "ResNet18-UNet",
            "loss_function": "BCE + Dice Loss",
            "epochs": epochs,
            "batch_size": batch_size,
            "min_component_area": min_component_area,
            "training_samples_count": len(train_samples),
            "training_duration_seconds": round(total_train_sec, 2),
            "best_nominal_val_dice": round(best_val_dice, 4),
            "primary_ground_truth_rule": "mask >= 128 with right border (x >= 477) zeroed out",
            "secondary_ground_truth_rule": "mask > 0 with right border (x >= 477) zeroed out",
            "prediction_threshold": 0.5,
            "environment_and_reproducibility": get_environment_info(),
        },
        "nominal_val": val_summary,
        "nominal_test": test_summary,
        "class6_val": c6_val_summary,
        "class6_test": c6_test_summary,
        "rare_holdout_summary": rare_summary,
        "rare_holdout_per_image": rare_records,
    }

    # 9. Predicted Bounding Boxes Export for Gate 2 SAM Hand-off
    all_eval_records = val_records + test_records + c6_val_records + c6_test_records + rare_records
    records_by_id = {r["id"]: r for r in all_eval_records}

    gate0_boxes = []
    for gid in gate0_ids:
        if gid in records_by_id:
            r = records_by_id[gid]
            gate0_boxes.append({
                "id": gid,
                "class_label": r["class_label"],
                "split": r["split"],
                "overall_bbox": r["overall_bbox"],
                "largest_component_bbox": r["largest_component_bbox"],
                "filtered_foreground_bbox": r["filtered_foreground_bbox"],
                "component_boxes": r["component_boxes"],
                "predicted_pixels": r["predicted_pixels"],
                "filtered_pixels": r["filtered_pixels"],
                "max_prob": r["max_prob"],
                "iou_primary": r["iou_primary"],
                "dice_primary": r["dice_primary"],
            })

    boxes_export = {
        "metadata": {
            "source_model": "ResNet18-UNet Baseline (Gate 1)",
            "prediction_prob_threshold": 0.5,
            "bbox_convention": "inclusive [x1, y1, x2, y2]",
            "border_zeroed": "x >= 477",
            "min_component_area": min_component_area,
            "intended_usage": "Non-Oracle Box Prompts for Meta SAM-3.1 Evaluation in Gate 1 Part B & Gate 2",
            "environment": get_environment_info(),
        },
        "gate0_20_predicted_boxes": gate0_boxes,
        "nominal_val_boxes": [
            {
                "id": r["id"],
                "class_label": r["class_label"],
                "overall_bbox": r["overall_bbox"],
                "largest_component_bbox": r["largest_component_bbox"],
                "filtered_foreground_bbox": r["filtered_foreground_bbox"],
                "component_boxes": r["component_boxes"],
                "predicted_pixels": r["predicted_pixels"],
                "max_prob": r["max_prob"],
                "iou_primary": r["iou_primary"],
            }
            for r in val_records
        ],
        "nominal_test_boxes": [
            {
                "id": r["id"],
                "class_label": r["class_label"],
                "overall_bbox": r["overall_bbox"],
                "largest_component_bbox": r["largest_component_bbox"],
                "filtered_foreground_bbox": r["filtered_foreground_bbox"],
                "component_boxes": r["component_boxes"],
                "predicted_pixels": r["predicted_pixels"],
                "max_prob": r["max_prob"],
                "iou_primary": r["iou_primary"],
            }
            for r in test_records
        ],
        "rare_holdout_boxes": [
            {
                "id": r["id"],
                "class_label": r["class_label"],
                "seen_in_gate0": r["seen_in_gate0"],
                "overall_bbox": r["overall_bbox"],
                "largest_component_bbox": r["largest_component_bbox"],
                "filtered_foreground_bbox": r["filtered_foreground_bbox"],
                "component_boxes": r["component_boxes"],
                "predicted_pixels": r["predicted_pixels"],
                "max_prob": r["max_prob"],
                "iou_primary": r["iou_primary"],
            }
            for r in rare_records
        ],
    }

    return best_weights_bytes, results, boxes_export


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate U-Net baseline for Carinthia-S.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--min-area", type=int, default=16, help="Minimum connected component area (default: 16 pixels).")
    args = parser.parse_args()

    print("=" * 80)
    print("GATE 1: U-NET SEGMENTATION BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    if not SPLITS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {SPLITS_MANIFEST_PATH}. Run scripts/create_gate1_splits.py first.")

    with open(SPLITS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        splits_manifest = json.load(f)
    splits_dict = splits_manifest["splits"]

    with open(GATE0_MANIFEST_PATH, "r", encoding="utf-8") as f:
        gate0_data = json.load(f)
    gate0_ids = [s["id"] for s in gate0_data["samples"]]

    print(f"Loaded splits manifest from {SPLITS_MANIFEST_PATH}")
    print(f"Splits: {list(splits_dict.keys())}")
    print(f"Loaded {len(gate0_ids)} Gate 0 IDs")

    data_root = PROJECT_ROOT / "data" / "carinthia_s"
    print(f"\n[Execution Mode] Training on local CUDA GPU using dataset at {data_root}...")
    checkpoint_bytes, results, boxes_export = run_training_and_evaluation(
        data_root_str=str(data_root),
        splits_dict=splits_dict,
        gate0_ids=gate0_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_component_area=args.min_area,
    )

    # 3. Save Checkpoint to gitignored outputs/gate1/checkpoints/
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / "best_unet.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    ckpt_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"\n[Artifact] Saved best U-Net checkpoint to: {checkpoint_path} ({ckpt_size_mb:.2f} MB)")

    # 4. Save Evaluation Results to results/gate1/unet_baseline_results.json (<1 MB)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "unet_baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    res_size_kb = results_path.stat().st_size / 1024
    print(f"[Artifact] Saved evaluation results to: {results_path} ({res_size_kb:.1f} KB)")

    # 5. Save Predicted Boxes to results/gate1/unet_predicted_boxes.json (<1 MB)
    boxes_path = RESULTS_DIR / "unet_predicted_boxes.json"
    with open(boxes_path, "w", encoding="utf-8") as f:
        json.dump(boxes_export, f, indent=2)
    boxes_size_kb = boxes_path.stat().st_size / 1024
    print(f"[Artifact] Saved predicted bounding boxes to: {boxes_path} ({boxes_size_kb:.1f} KB)")

    # 6. Verification Summary
    print("\n" + "=" * 80)
    print("GATE 1 U-NET BASELINE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Training Duration: {results['metadata']['training_duration_seconds']}s across {results['metadata']['epochs']} epochs")
    print(f"Best nominal_val Dice: {results['metadata']['best_nominal_val_dice']:.4f}")
    print(f"Environment: {results['metadata']['environment_and_reproducibility']['cuda_device_name']} | PyTorch {results['metadata']['environment_and_reproducibility']['torch_version']}")

    print("\n[Nominal Validation (430 images)]")
    nv = results["nominal_val"]
    print(f"  Primary (mask >= 128): Median IoU = {nv['median_iou_primary']:.4f} | Mean IoU = {nv['mean_iou_primary']:.4f} | Mean Dice = {nv['mean_dice_primary']:.4f}")
    print(f"  Secondary (mask > 0):  Median IoU = {nv['median_iou_secondary']:.4f} | Mean IoU = {nv['mean_iou_secondary']:.4f} | Mean Dice = {nv['mean_dice_secondary']:.4f}")
    for c_key, c_val in nv["per_class"].items():
        print(f"    - {c_key} ({c_val['count']} imgs): Primary Median IoU = {c_val['median_iou_primary']:.4f}, Mean IoU = {c_val['mean_iou_primary']:.4f}, Mean Dice = {c_val['mean_dice_primary']:.4f}")

    print("\n[Nominal Test (430 unseen images)]")
    nt = results["nominal_test"]
    print(f"  Primary (mask >= 128): Median IoU = {nt['median_iou_primary']:.4f} | Mean IoU = {nt['mean_iou_primary']:.4f} | Mean Dice = {nt['mean_dice_primary']:.4f}")
    print(f"  Secondary (mask > 0):  Median IoU = {nt['median_iou_secondary']:.4f} | Mean IoU = {nt['mean_iou_secondary']:.4f} | Mean Dice = {nt['mean_dice_secondary']:.4f}")
    for c_key, c_val in nt["per_class"].items():
        print(f"    - {c_key} ({c_val['count']} imgs): Primary Median IoU = {c_val['median_iou_primary']:.4f}, Mean IoU = {c_val['mean_iou_primary']:.4f}, Mean Dice = {c_val['mean_dice_primary']:.4f}")

    print("\n[Class 6 (Empty-Scan Misalignment / False Alarm)]")
    c6_t = results["class6_test"]
    if "empty_scan_metrics" in c6_t:
        esm = c6_t["empty_scan_metrics"]
        print(f"  Empty-Scan Specificity: {esm['empty_scan_specificity']}% ({esm['empty_scans_zero_predicted']}/{esm['empty_scans_total']} empty scans with 0 predicted FG pixels)")
        print(f"  False-Positive Pixels:  Mean = {esm['mean_false_positive_pixels']}, Median = {esm['median_false_positive_pixels']}, Max = {esm['max_false_positive_pixels']}")

    print("\n[Rare Holdout (67 images - Classes 1, 2, 5)]")
    rh = results["rare_holdout_summary"]
    print(f"  Overall: Primary Median IoU = {rh['median_iou_primary']:.4f} | Mean IoU = {rh['mean_iou_primary']:.4f} | Mean Dice = {rh['mean_dice_primary']:.4f}")
    for c_key, c_val in rh["per_class"].items():
        print(f"    - {c_key} ({c_val['count']} imgs): Primary Median IoU = {c_val['median_iou_primary']:.4f}, Mean IoU = {c_val['mean_iou_primary']:.4f}, Mean Dice = {c_val['mean_dice_primary']:.4f}")

    print("\n[Gate 0 20-Sample Non-Oracle Predicted Bounding Boxes]")
    g0_boxes = boxes_export["gate0_20_predicted_boxes"]
    valid_boxes = sum(1 for b in g0_boxes if b["largest_component_bbox"] is not None)
    print(f"  Extracted largest-component boxes for {valid_boxes}/{len(g0_boxes)} Gate 0 images (ready for SAM 3.1 prompt input).")
    print(f"  Total rare holdout boxes exported: {len(boxes_export['rare_holdout_boxes'])}/67")
    print("=" * 80)


if __name__ == "__main__":
    main()
