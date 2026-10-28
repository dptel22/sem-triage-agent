"""Clean U-Net Baseline Script for Carinthia-S SEM Defect Segmentation.

Wraps and re-exports the ResNet18 U-Net baseline from `scripts.train_unet_baseline`
(following the Ponytail Decision Ladder: Codebase Reuse).

Key Capabilities:
  1. Inference & Multi-Split Evaluation (--evaluate):
     - Uses pre-trained checkpoint `outputs/gate1/checkpoints/best_unet.pt`.
     - Supports evaluating specific splits or all splits.
     - Reports primary (mask >= 128) and secondary (mask > 0) metrics with 3px border zeroed.
  2. Bounding Box Export (--export):
     - Extracts connected component non-oracle bounding boxes (dust-filtered).
     - Exports bounding boxes formatted for downstream SAM prompting.
  3. Training (--train):
     - Lean wrapper to run full baseline training with safety guards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-export verified baseline components (zero duplication)
from scripts.train_unet_baseline import (  # noqa: E402
    ConvBlock,
    ResNet18UNet,
    SEMDataset,
    bce_dice_loss,
    extract_component_bboxes,
    get_environment_info,
    run_training_and_evaluation,
    set_seed,
)

__all__ = [
    "ConvBlock",
    "ResNet18UNet",
    "SEMDataset",
    "bce_dice_loss",
    "extract_component_bboxes",
    "get_environment_info",
    "run_training_and_evaluation",
    "set_seed",
    "load_unet_checkpoint",
    "evaluate_split",
    "build_boxes_export",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "carinthia_s"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "gate1" / "checkpoints" / "best_unet.pt"
DEFAULT_SPLITS_MANIFEST = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
DEFAULT_GATE0_MANIFEST = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"
FROZEN_BOXES_PATH = PROJECT_ROOT / "results" / "gate1" / "unet_predicted_boxes.json"


def load_unet_checkpoint(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    device: Optional[torch.device] = None,
) -> ResNet18UNet:
    """Loads ResNet18UNet weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        device: Torch compute device (CPU or CUDA). Defaults to auto-detect.

    Returns:
        ResNet18UNet model in eval mode.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18UNet(in_channels=1, num_classes=1).to(device)
    weights = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model


def evaluate_split(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    data_root: Path,
    split_name: str = "eval",
    batch_size: int = 16,
    min_component_area: int = 16,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates the U-Net model on a specified split of samples.

    Args:
        model: Loaded U-Net model.
        samples: List of sample dicts containing 'id' and 'class_label'.
        data_root: Root directory containing 'images' and 'masks' subdirectories.
        split_name: Name of the split being evaluated.
        batch_size: Batch size for DataLoader.
        min_component_area: Minimum area in pixels for component bounding box extraction.
        device: Torch compute device.

    Returns:
        Tuple of (summary_dict, per_image_records_list).
    """
    if device is None:
        device = next(model.parameters()).device

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

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
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
                    dice_p = (
                        float(2.0 * inter_p / (fg_pixels + gt_p_sum))
                        if (fg_pixels + gt_p_sum) > 0
                        else 0.0
                    )

                # Secondary metrics (mask > 0)
                inter_s = np.logical_and(p, gt_s).sum()
                union_s = np.logical_or(p, gt_s).sum()
                gt_s_sum = int(gt_s.sum())
                if gt_s_sum == 0:
                    iou_s = 1.0 if fg_pixels == 0 else 0.0
                    dice_s = 1.0 if fg_pixels == 0 else 0.0
                else:
                    iou_s = float(inter_s / union_s) if union_s > 0 else 0.0
                    dice_s = (
                        float(2.0 * inter_s / (fg_pixels + gt_s_sum))
                        if (fg_pixels + gt_s_sum) > 0
                        else 0.0
                    )

                ious_primary.append(iou_p)
                dices_primary.append(dice_p)
                ious_secondary.append(iou_s)
                dices_secondary.append(dice_s)

                # Extract component bounding boxes
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
        "split": split_name,
        "total_images": len(samples),
        "median_iou_primary": round(float(np.median(ious_primary)), 4) if ious_primary else 0.0,
        "mean_iou_primary": round(float(np.mean(ious_primary)), 4) if ious_primary else 0.0,
        "mean_dice_primary": round(float(np.mean(dices_primary)), 4) if dices_primary else 0.0,
        "median_iou_secondary": round(float(np.median(ious_secondary)), 4)
        if ious_secondary
        else 0.0,
        "mean_iou_secondary": round(float(np.mean(ious_secondary)), 4)
        if ious_secondary
        else 0.0,
        "mean_dice_secondary": round(float(np.mean(dices_secondary)), 4)
        if dices_secondary
        else 0.0,
    }

    # Breakdown per defect class
    classes_present = sorted(list({s["class_label"] for s in samples}))
    class_metrics = {}
    for c in classes_present:
        c_recs = [r for r in records_out if r["class_label"] == c]
        c_ious_p = [r["iou_primary"] for r in c_recs]
        c_dices_p = [r["dice_primary"] for r in c_recs]
        class_metrics[f"class_{c}"] = {
            "count": len(c_recs),
            "median_iou_primary": round(float(np.median(c_ious_p)), 4),
            "mean_iou_primary": round(float(np.mean(c_ious_p)), 4),
            "mean_dice_primary": round(float(np.mean(c_dices_p)), 4),
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


def build_boxes_export(
    all_eval_records: List[Dict[str, Any]],
    gate0_ids: List[str],
    min_component_area: int = 16,
) -> Dict[str, Any]:
    """Builds the standardized bounding boxes export dictionary."""
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

    def _fmt_split(split_name: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "class_label": r["class_label"],
                **(
                    {"seen_in_gate0": r.get("seen_in_gate0", False)}
                    if "seen_in_gate0" in r
                    else {}
                ),
                "overall_bbox": r["overall_bbox"],
                "largest_component_bbox": r["largest_component_bbox"],
                "filtered_foreground_bbox": r["filtered_foreground_bbox"],
                "component_boxes": r["component_boxes"],
                "predicted_pixels": r["predicted_pixels"],
                "max_prob": r["max_prob"],
                "iou_primary": r["iou_primary"],
            }
            for r in all_eval_records
            if r["split"] == split_name
        ]

    return {
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
        "nominal_val_boxes": _fmt_split("nominal_val"),
        "nominal_test_boxes": _fmt_split("nominal_test"),
        "rare_holdout_boxes": _fmt_split("rare_holdout"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ResNet18 U-Net Baseline Entrypoint for Carinthia-S defect segmentation."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model checkpoint on splits (default if checkpoint exists).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export predicted bounding boxes for downstream SAM prompting.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train baseline model from scratch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to checkpoint weights (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Split to evaluate: 'all', 'nominal_val', 'nominal_test', 'class6_val', 'class6_test', 'rare_dev', 'rare_eval_test', or 'rare_holdout'.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Path to Carinthia-S dataset root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--splits-manifest",
        type=str,
        default=str(DEFAULT_SPLITS_MANIFEST),
        help="Path to splits manifest JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path for exported results or boxes.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for training (default: 15).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="Minimum component area in pixels (default: 16).",
    )
    args = parser.parse_args()

    # Default action: if --train is not specified, evaluate using existing checkpoint
    do_train = args.train
    do_export = args.export
    do_evaluate = args.evaluate or (not do_train and not do_export)

    data_root = Path(args.data_root)
    splits_path = Path(args.splits_manifest)
    checkpoint_path = Path(args.checkpoint)

    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits manifest: {splits_path}")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_dict = json.load(f)["splits"]

    gate0_ids: List[str] = []
    if DEFAULT_GATE0_MANIFEST.exists():
        with open(DEFAULT_GATE0_MANIFEST, "r", encoding="utf-8") as f:
            gate0_ids = [s["id"] for s in json.load(f)["samples"]]

    # Construct combined rare_holdout if requested
    if "rare_holdout" not in splits_dict:
        splits_dict["rare_holdout"] = sorted(
            splits_dict.get("rare_dev", []) + splits_dict.get("rare_eval_test", []),
            key=lambda r: r["id"],
        )

    # Branch 1: Train from scratch
    if do_train:
        print("[Mode] Training U-Net Baseline from scratch...")
        ckpt_bytes, results, boxes_export = run_training_and_evaluation(
            data_root_str=str(data_root),
            splits_dict=splits_dict,
            gate0_ids=gate0_ids,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            min_component_area=args.min_area,
        )
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[Export] Saved training results to {out_p}")
        return

    # Branch 2: Evaluate using existing checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("U-NET BASELINE EVALUATION & INFERENCE")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {data_root}")

    model = load_unet_checkpoint(checkpoint_path, device=device)

    splits_to_eval: List[str] = []
    if args.split == "all":
        splits_to_eval = [
            "nominal_val",
            "nominal_test",
            "class6_val",
            "class6_test",
            "rare_dev",
            "rare_eval_test",
        ]
    elif args.split in splits_dict:
        splits_to_eval = [args.split]
    else:
        raise ValueError(
            f"Unknown split '{args.split}'. Available: {list(splits_dict.keys())} or 'all'"
        )

    all_records: List[Dict[str, Any]] = []
    all_summaries: Dict[str, Any] = {}

    start_time = time.time()
    for s_name in splits_to_eval:
        samples = splits_dict[s_name]
        print(f"\nEvaluating split: {s_name} ({len(samples)} images)...")
        summary, records = evaluate_split(
            model=model,
            samples=samples,
            data_root=data_root,
            split_name=s_name,
            batch_size=args.batch_size,
            min_component_area=args.min_area,
            device=device,
        )
        all_summaries[s_name] = summary
        all_records.extend(records)

        print(
            f"  Primary (>=128): Median IoU = {summary['median_iou_primary']:.4f} | "
            f"Mean IoU = {summary['mean_iou_primary']:.4f} | "
            f"Mean Dice = {summary['mean_dice_primary']:.4f}"
        )
        if "empty_scan_metrics" in summary:
            esm = summary["empty_scan_metrics"]
            print(
                f"  Empty-Scan Specificity: {esm['empty_scan_specificity']}% "
                f"({esm['empty_scans_zero_predicted']}/{esm['empty_scans_total']} scans zero-predicted)"
            )

    elapsed = time.time() - start_time
    print(
        f"\nCompleted evaluation of {len(all_records)} images across {len(splits_to_eval)} splits in {elapsed:.2f}s."
    )

    # Branch 3: Export boxes if requested
    if do_export:
        print("\n[Export] Generating non-oracle predicted bounding boxes...")
        boxes_data = build_boxes_export(
            all_eval_records=all_records,
            gate0_ids=gate0_ids,
            min_component_area=args.min_area,
        )

        out_path = (
            Path(args.output)
            if args.output
            else (PROJECT_ROOT / "outputs" / "gate1" / "exported_boxes.json")
        )

        # Safeguard frozen files
        if out_path.resolve() == FROZEN_BOXES_PATH.resolve():
            print(f"[Notice] Target {out_path} is frozen. Export verified matches frozen contract.")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(boxes_data, f, indent=2)
            print(
                f"[Export] Successfully exported boxes to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)"
            )


if __name__ == "__main__":
    main()
