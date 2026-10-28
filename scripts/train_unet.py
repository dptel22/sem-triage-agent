"""Train and evaluate baseline ResNet18-UNet for SEM defect proposal.

Usage:
  python scripts/train_unet.py --evaluate --split nominal_val
  python scripts/train_unet.py --export-boxes
  python scripts/train_unet.py --train --epochs 15 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import preprocess_mask  # noqa: E402
from src.unet import ResNet18UNet, extract_component_bboxes, load_unet, predict_prob  # noqa: E402


def evaluate_split(
    model: ResNet18UNet,
    split_name: str,
    samples: List[Dict[str, Any]],
    data_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates U-Net IoU and pixel metrics on a dataset split."""
    model.eval()
    primary_ious = []
    secondary_ious = []
    fg_pixels = []

    for s in samples:
        sid = s["id"]
        img_p = data_dir / "images" / f"{sid}.jpg"
        mask_p = data_dir / "masks" / f"{sid}.png"

        if not img_p.exists() or not mask_p.exists():
            continue

        img = Image.open(img_p).convert("L")
        prob = predict_prob(model, img, device=device)
        pred_bin = (prob >= 0.5).astype(np.uint8)

        gt_raw = np.array(Image.open(mask_p).convert("L"))
        gt_pri = preprocess_mask(gt_raw, threshold=128, zero_border=True)
        gt_sec = preprocess_mask(gt_raw, threshold=0, zero_border=True)

        # Primary IoU
        inter_pri = np.logical_and(pred_bin > 0, gt_pri > 0).sum()
        union_pri = np.logical_or(pred_bin > 0, gt_pri > 0).sum()
        iou_pri = float(inter_pri / union_pri) if union_pri > 0 else (1.0 if pred_bin.sum() == 0 else 0.0)

        # Secondary IoU
        inter_sec = np.logical_and(pred_bin > 0, gt_sec > 0).sum()
        union_sec = np.logical_or(pred_bin > 0, gt_sec > 0).sum()
        iou_sec = float(inter_sec / union_sec) if union_sec > 0 else (1.0 if pred_bin.sum() == 0 else 0.0)

        primary_ious.append(iou_pri)
        secondary_ious.append(iou_sec)
        fg_pixels.append(int(pred_bin.sum()))

    return {
        "split": split_name,
        "n_samples": len(primary_ious),
        "mean_iou_primary": float(np.mean(primary_ious)) if primary_ious else 0.0,
        "median_iou_primary": float(np.median(primary_ious)) if primary_ious else 0.0,
        "mean_iou_secondary": float(np.mean(secondary_ious)) if secondary_ious else 0.0,
        "median_iou_secondary": float(np.median(secondary_ious)) if secondary_ious else 0.0,
        "mean_predicted_fg_px": float(np.mean(fg_pixels)) if fg_pixels else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train / evaluate ResNet18-UNet baseline.")
    parser.add_argument("--train", action="store_true", help="Train model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model.")
    parser.add_argument("--export-boxes", action="store_true", help="Export predicted bounding boxes.")
    parser.add_argument("--split", type=str, default="nominal_val", help="Split to evaluate.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--checkpoint", type=str, default="outputs/gate1/checkpoints/best_unet.pt", help="Checkpoint path.")
    parser.add_argument("--data-dir", type=str, default="data/carinthia_s", help="Path to carinthia_s data dir.")
    parser.add_argument("--manifest", type=str, default="results/gate1/splits_manifest.json", help="Path to splits manifest.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    splits_data = json.load(open(manifest_path, encoding="utf-8"))["splits"]

    if args.evaluate:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"Error: Checkpoint not found at {ckpt}", file=sys.stderr)
            sys.exit(1)

        model = load_unet(ckpt, device=device)
        print(f"Loaded checkpoint {ckpt} onto {device}")

        splits_to_eval = [args.split] if args.split != "all" else list(splits_data.keys())
        for sp in splits_to_eval:
            if sp not in splits_data:
                print(f"Split {sp} not found in manifest")
                continue
            metrics = evaluate_split(model, sp, splits_data[sp], data_dir, device)
            print(f"\n--- {sp} ({metrics['n_samples']} samples) ---")
            print(f"  Primary Median IoU:   {metrics['median_iou_primary']:.4f}")
            print(f"  Primary Mean IoU:     {metrics['mean_iou_primary']:.4f}")
            print(f"  Secondary Median IoU: {metrics['median_iou_secondary']:.4f}")
            print(f"  Mean Pred FG Pixels:  {metrics['mean_predicted_fg_px']:.1f}")

    elif args.export_boxes:
        ckpt = Path(args.checkpoint)
        model = load_unet(ckpt, device=device)
        print(f"Exporting non-oracle predicted bounding boxes using {ckpt}...")

        all_boxes: Dict[str, Any] = {"metadata": {"source_checkpoint": str(ckpt)}}
        for sp_name, samples in splits_data.items():
            sp_boxes = []
            for s in samples:
                sid = s["id"]
                img_p = data_dir / "images" / f"{sid}.jpg"
                if not img_p.exists():
                    continue
                img = Image.open(img_p).convert("L")
                prob = predict_prob(model, img, device=device)
                bin_mask = (prob >= 0.5).astype(np.uint8)
                comp = extract_component_bboxes(bin_mask, min_area=16)
                sp_boxes.append({
                    "id": sid,
                    "class_label": s.get("class_label", s.get("label", -1)),
                    "largest_component_bbox": comp["largest_component_bbox"],
                    "filtered_foreground_bbox": comp["filtered_foreground_bbox"],
                    "overall_bbox": comp["overall_bbox"],
                    "component_boxes": comp["component_boxes"],
                    "predicted_foreground_pixels_filtered": comp["filtered_pixel_count"],
                    "predicted_foreground_pixels_raw": int(bin_mask.sum()),
                    "max_prob": float(prob.max()),
                })
            all_boxes[f"{sp_name}_predicted_boxes"] = sp_boxes
            print(f"  {sp_name}: {len(sp_boxes)} boxes extracted")

        out_path = PROJECT_ROOT / "outputs" / "gate1" / "recomputed_predicted_boxes.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_boxes, indent=2), encoding="utf-8")
        print(f"Exported boxes to {out_path}")

    else:
        print("Specify --evaluate or --export-boxes. (Training requires local dataset).")


if __name__ == "__main__":
    main()
