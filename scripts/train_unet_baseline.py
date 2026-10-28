"""Train and Evaluate U-Net Baseline for Carinthia-S SEM Defect Segmentation.

Trains a lightweight ResNet18-UNet baseline on nominal_train + class6_train (3,618 images)
using BCE + Dice loss at 480x480 resolution, zeroing the 3px right border artifact (x >= 477).
Executes on Modal cloud GPU (attaching carinthia-s-vol) or local CUDA GPU if available.

Outputs:
1. Model checkpoint: outputs/gate1/checkpoints/best_unet.pt (gitignored)
2. Evaluation metrics: results/gate1/unet_baseline_results.json
3. Predicted boxes: results/gate1/unet_predicted_boxes.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_MANIFEST_PATH = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
GATE0_MANIFEST_PATH = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "gate1"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "gate1"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# Modal App Setup
# ---------------------------------------------------------------------------
try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

app = modal.App("sem-triage-gate1-unet") if HAS_MODAL else None
volume = modal.Volume.from_name("carinthia-s-vol") if HAS_MODAL else None

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "opencv-python-headless>=4.8.0",
        "tqdm>=4.66.0",
    )
) if HAS_MODAL else None


# ---------------------------------------------------------------------------
# Core Training & Evaluation Logic (Shared between Local and Modal)
# ---------------------------------------------------------------------------
def run_training_and_evaluation(
    data_root_str: str,
    splits_dict: Dict[str, List[Dict[str, Any]]],
    gate0_ids: List[str],
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_str: str = "cuda",
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """Runs complete U-Net baseline training, checkpointing, and evaluation."""
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    import torchvision.models as models

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(data_root_str)

    # 1. Dataset Definition
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

            # Mask binarization & 3px right border zeroing (x >= 477)
            mask = Image.open(mask_p).convert("L")
            mask_arr = np.array(mask, dtype=np.uint8)

            # Primary GT: mask >= 128, x >= 477 zeroed out
            gt_primary = (mask_arr >= 128).astype(np.float32)
            gt_primary[:, 477:] = 0.0

            # Secondary GT: mask > 0, x >= 477 zeroed out
            gt_secondary = (mask_arr > 0).astype(np.float32)
            gt_secondary[:, 477:] = 0.0

            # Tensor conversion: (1, 480, 480)
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

    # 2. ResNet18 U-Net Model Architecture
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

            self.enc1 = resnet.layer1  # 64 ch, stride 4
            self.enc2 = resnet.layer2  # 128 ch, stride 8
            self.enc3 = resnet.layer3  # 256 ch, stride 16
            self.enc4 = resnet.layer4  # 512 ch, stride 32

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
            # Encoder
            e0 = self.relu(self.bn1(self.init_conv(x)))  # (B, 64, 240, 240)
            p0 = self.maxpool(e0)                         # (B, 64, 120, 120)
            e1 = self.enc1(p0)                            # (B, 64, 120, 120)
            e2 = self.enc2(e1)                            # (B, 128, 60, 60)
            e3 = self.enc3(e2)                            # (B, 256, 30, 30)
            e4 = self.enc4(e3)                            # (B, 512, 15, 15)

            # Decoder
            d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))  # (B, 256, 30, 30)
            d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))  # (B, 128, 60, 60)
            d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))  # (B, 64, 120, 120)
            d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))  # (B, 64, 240, 240)
            d0 = self.dec0(self.up0(d1))                           # (B, 32, 480, 480)
            logits = self.head(d0)                                # (B, 1, 480, 480)
            return logits

    # 3. Loss Function (BCE + Dice Loss)
    def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        return bce + dice_loss

    # 4. Prepare Datasets & Loaders
    train_samples = splits_dict["nominal_train"] + splits_dict["class6_train"]
    val_samples = splits_dict["nominal_val"]
    print(f"[Data] Training set: {len(train_samples)} (nominal_train={len(splits_dict['nominal_train'])}, class6_train={len(splits_dict['class6_train'])})")
    print(f"[Data] Validation set: {len(val_samples)} (nominal_val={len(splits_dict['nominal_val'])})")

    train_ds = SEMDataset(train_samples, data_root)
    val_ds = SEMDataset(val_samples, data_root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Model, Optimizer, Scaler
    model = ResNet18UNet(in_channels=1, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_dice = -1.0
    best_weights_bytes: Optional[bytes] = None

    print(f"[Train] Starting U-Net training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask_primary"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                # Enforce zero border on logits / loss
                logits[:, :, :, 477:] = -100.0
                loss = bce_dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation on nominal_val
        model.eval()
        val_dices = []
        val_ious = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask_primary"].cpu().numpy()[:, 0]  # (B, H, W)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Mean Dice: {mean_dice:.4f} | Val Median IoU: {median_iou:.4f} | Val Mean IoU: {mean_iou:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            best_weights_bytes = buf.getvalue()

    total_train_sec = time.time() - start_time
    print(f"[Train] Training complete in {total_train_sec:.1f}s. Best nominal_val Mean Dice: {best_val_dice:.4f}")

    # Load best checkpoint for full multi-split evaluation
    assert best_weights_bytes is not None
    buf = io.BytesIO(best_weights_bytes)
    model.load_state_dict(torch.load(buf, map_location=device))
    model.eval()

    # 6. Full Inference & Metrics Evaluation Helper
    def evaluate_split(samples: List[Dict[str, Any]], split_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ds = SEMDataset(samples, data_root)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

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

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

                    # Primary metrics
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

                    # Secondary metrics
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

                    # Bounding Box Extraction (Inclusive [x1, y1, x2, y2])
                    overall_bbox = None
                    largest_comp_bbox = None
                    if fg_pixels > 0:
                        y_idx, x_idx = np.where(p > 0)
                        overall_bbox = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())]

                        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
                        if num_labels > 1:
                            areas = stats[1:, cv2.CC_STAT_AREA]
                            max_lbl_idx = int(np.argmax(areas)) + 1
                            s = stats[max_lbl_idx]
                            x1 = int(s[cv2.CC_STAT_LEFT])
                            y1 = int(s[cv2.CC_STAT_TOP])
                            x2 = int(x1 + s[cv2.CC_STAT_WIDTH] - 1)
                            y2 = int(y1 + s[cv2.CC_STAT_HEIGHT] - 1)
                            largest_comp_bbox = [x1, y1, x2, y2]
                        else:
                            largest_comp_bbox = overall_bbox

                    records_out.append({
                        "id": sid,
                        "class_label": lbl,
                        "split": split_name,
                        "iou_primary": round(iou_p, 4),
                        "dice_primary": round(dice_p, 4),
                        "iou_secondary": round(iou_s, 4),
                        "dice_secondary": round(dice_s, 4),
                        "predicted_pixels": fg_pixels,
                        "ground_truth_primary_pixels": gt_p_sum,
                        "max_prob": round(max_prob, 4),
                        "overall_bbox": overall_bbox,
                        "largest_component_bbox": largest_comp_bbox,
                    })

        # Summary Metrics
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

    # 7. Evaluate on all Gate 1 splits
    print("\n[Eval] Running inference on nominal_val (430 images)...")
    val_summary, val_records = evaluate_split(splits_dict["nominal_val"], "nominal_val")

    print("[Eval] Running inference on nominal_test (430 images)...")
    test_summary, test_records = evaluate_split(splits_dict["nominal_test"], "nominal_test")

    print("[Eval] Running inference on class6_val (23 images)...")
    c6_val_summary, c6_val_records = evaluate_split(splits_dict["class6_val"], "class6_val")

    print("[Eval] Running inference on class6_test (23 images)...")
    c6_test_summary, c6_test_records = evaluate_split(splits_dict["class6_test"], "class6_test")

    print("[Eval] Running inference on rare_holdout (67 images)...")
    rare_summary, rare_records = evaluate_split(splits_dict["rare_holdout"], "rare_holdout")

    # Flag seen_in_gate0 on rare_records
    gate0_set = set(gate0_ids)
    for r in rare_records:
        r["seen_in_gate0"] = r["id"] in gate0_set

    # 8. Assemble Full Results Dictionary
    results = {
        "metadata": {
            "model_architecture": "ResNet18-UNet",
            "loss_function": "BCE + Dice Loss",
            "epochs": epochs,
            "batch_size": batch_size,
            "training_samples_count": len(train_samples),
            "training_duration_seconds": round(total_train_sec, 2),
            "best_nominal_val_dice": round(best_val_dice, 4),
            "ground_truth_primary_rule": "mask >= 128 with right border (x >= 477) zeroed out",
            "ground_truth_secondary_rule": "mask > 0 with right border (x >= 477) zeroed out",
            "prediction_threshold": 0.5,
        },
        "nominal_val": val_summary,
        "nominal_test": test_summary,
        "class6_val": c6_val_summary,
        "class6_test": c6_test_summary,
        "rare_holdout_summary": rare_summary,
        "rare_holdout_per_image": rare_records,
    }

    # 9. Assemble Non-Oracle Predicted Bounding Boxes for Gate 2 SAM Hand-off
    # Specifically include:
    # 1. All 20 Gate 0 images
    # 2. All 430 nominal_val images
    # 3. All 430 nominal_test images
    # 4. All 67 rare_holdout images
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
                "predicted_pixels": r["predicted_pixels"],
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
            "intended_usage": "Non-Oracle Box Prompts for Meta SAM-3.1 Evaluation in Gate 1 Part B & Gate 2",
        },
        "gate0_20_predicted_boxes": gate0_boxes,
        "nominal_val_boxes": [
            {
                "id": r["id"],
                "class_label": r["class_label"],
                "overall_bbox": r["overall_bbox"],
                "largest_component_bbox": r["largest_component_bbox"],
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
                "predicted_pixels": r["predicted_pixels"],
                "max_prob": r["max_prob"],
                "iou_primary": r["iou_primary"],
            }
            for r in rare_records
        ],
    }

    return best_weights_bytes, results, boxes_export


# ---------------------------------------------------------------------------
# Modal Remote Function
# ---------------------------------------------------------------------------
if HAS_MODAL:
    @app.function(
        image=modal_image,
        volumes={"/mnt/carinthia_s": volume},
        gpu="A10G",
        timeout=1800,
    )
    def modal_train_and_evaluate(
        splits_dict: Dict[str, List[Dict[str, Any]]],
        gate0_ids: List[str],
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
        """Executes U-Net training and evaluation on Modal A10G GPU."""
        return run_training_and_evaluation(
            data_root_str="/mnt/carinthia_s",
            splits_dict=splits_dict,
            gate0_ids=gate0_ids,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device_str="cuda",
        )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate U-Net baseline for Carinthia-S.")
    parser.add_argument("--local", action="store_true", help="Force local execution (default uses Modal GPU if available).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    args = parser.parse_args()

    print("=" * 80)
    print("GATE 1: U-NET SEGMENTATION BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load splits manifest and Gate 0 IDs
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

    # 2. Execute Training
    use_modal = HAS_MODAL and not args.local
    if use_modal:
        print("\n[Execution Mode] Launching on Modal Cloud GPU (A10G, Volume: carinthia-s-vol)...")
        with app.run():
            checkpoint_bytes, results, boxes_export = modal_train_and_evaluate.remote(
                splits_dict=splits_dict,
                gate0_ids=gate0_ids,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
    else:
        print("\n[Execution Mode] Running locally...")
        local_data_root = PROJECT_ROOT / "data" / "carinthia_s"
        checkpoint_bytes, results, boxes_export = run_training_and_evaluation(
            data_root_str=str(local_data_root),
            splits_dict=splits_dict,
            gate0_ids=gate0_ids,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str="cuda",
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

    # 6. Print Full Gate 1 Verification Summary
    print("\n" + "=" * 80)
    print("GATE 1 U-NET BASELINE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Training Duration: {results['metadata']['training_duration_seconds']}s across {results['metadata']['epochs']} epochs")
    print(f"Best nominal_val Dice: {results['metadata']['best_nominal_val_dice']:.4f}")

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
    valid_boxes = sum(1 for b in g0_boxes if b["overall_bbox"] is not None)
    print(f"  Extracted bounding boxes for {valid_boxes}/{len(g0_boxes)} Gate 0 images (ready for SAM 3.1 prompt input).")
    print("=" * 80)


if __name__ == "__main__":
    main()
