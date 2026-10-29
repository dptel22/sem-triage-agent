"""Unified Evaluation Runner for WaferGuard SEM Defect Triage Agent.

Supports:
  - `--mode sharpened-10`: Evaluates the pre-registered 10-sample verification suite.
  - `--mode dev-dry-run`: Evaluates the 4-sample development suite (GT-free proxy verification).
  - `--mode full`: Evaluates across all rare_eval_test (50) and nominal_test (430) images.

Usage:
  python scripts/run_eval.py --mode sharpened-10
  python scripts/run_eval.py --mode dev-dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.triage_agent import (  # noqa: E402
    RetrievalBank,
    VerifiableTriageAgent,
    resolve_api_key,
)
from src.dataset import preprocess_mask  # noqa: E402


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection-over-Union between two binary masks."""
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return float(inter / union) if union > 0 else (1.0 if m1.sum() == 0 and m2.sum() == 0 else 0.0)


def run_evaluation(
    mode: str,
    output_path: Optional[Path] = None,
    triage_model: str = "muse-spark-1.3",
    api_base_url: Optional[str] = None,
    data_dir: Path | str = "data/carinthia_s",
) -> Dict[str, Any]:
    data_p = Path(data_dir)
    is_open_weights = "open" in triage_model.lower() or "baseten" in str(api_base_url or "").lower()
    default_base_url = os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1") if is_open_weights else "https://api.meta.ai/v1"
    base_url = api_base_url or default_base_url

    try:
        api_key = resolve_api_key(provider="baseten" if is_open_weights else "meta")
    except ValueError:
        api_key = None

    print("=" * 80)
    print(f"WAFERGUARD TRIAGE AGENT EVALUATION — MODE: {mode.upper()} — MODEL: {triage_model}")
    print("=" * 80)

    # 1. Load Precomputed Retrieval Bank
    bank_cache = PROJECT_ROOT / "outputs" / "gate3" / "retrieval_bank.json"
    bank = RetrievalBank.load_or_build(
        data_dir=data_p,
        manifest_path=PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json",
        cache_path=bank_cache,
    )

    # 2. Initialize Agent
    agent = VerifiableTriageAgent(
        retrieval_bank=bank,
        unet_model_path=PROJECT_ROOT / "outputs" / "gate1" / "checkpoints" / "best_unet.pt",
        triage_model=triage_model,
        api_base_url=base_url,
        api_key=api_key,
    )

    # 3. Load Manifests
    splits_manifest = json.load(open(PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json", encoding="utf-8"))["splits"]
    unet_boxes_data = json.load(open(PROJECT_ROOT / "results" / "gate1" / "unet_predicted_boxes.json", encoding="utf-8"))

    all_unet_entries = {}
    for key, entries in unet_boxes_data.items():
        if isinstance(entries, list):
            for e in entries:
                all_unet_entries[e["id"]] = e

    if mode in ("sharpened-10", "official-10"):
        manifest_path = PROJECT_ROOT / "results" / "gate1" / "sharpened_test_10_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            sh_data = json.load(f)
        eval_samples = sh_data["samples"]

    elif mode == "dev-dry-run":
        eval_samples = [
            {"id": splits_manifest["nominal_val"][0]["id"], "class_label": splits_manifest["nominal_val"][0]["class_label"], "role": "nominal_clean", "perturb": False},
            {"id": splits_manifest["nominal_val"][1]["id"], "class_label": splits_manifest["nominal_val"][1]["class_label"], "role": "nominal_perturbed", "perturb": True},
            {"id": splits_manifest["rare_dev"][0]["id"], "class_label": splits_manifest["rare_dev"][0]["class_label"], "role": "rare_clean", "perturb": False},
            {"id": splits_manifest["class6_val"][0]["id"], "class_label": splits_manifest["class6_val"][0]["class_label"], "role": "class6_empty", "perturb": False},
        ]
    elif mode == "dev-calibration":
        eval_samples = [
            {"id": s["id"], "class_label": s["class_label"], "role": "rare_dev", "perturb": False}
            for s in splits_manifest["rare_dev"]
        ] + [
            {"id": s["id"], "class_label": s["class_label"], "role": "nominal_val", "perturb": False}
            for s in splits_manifest["nominal_val"][:17]
        ]
        if output_path is None:
            output_path = PROJECT_ROOT / "outputs" / "gate3" / "dev_calibration.json"
    elif mode == "full":
        eval_samples = [
            {"id": s["id"], "class_label": s["class_label"], "role": "rare_eval_test", "perturb": False}
            for s in splits_manifest["rare_eval_test"]
        ] + [
            {"id": s["id"], "class_label": s["class_label"], "role": "nominal_test", "perturb": False}
            for s in splits_manifest["nominal_test"]
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results_list: List[Dict[str, Any]] = []

    for item in eval_samples:
        sid = item["id"]
        role = item.get("role", "test_sample")
        perturb = item.get("perturb", False)
        lbl = item.get("class_label", item.get("class", -1))

        img_path = data_p / "images" / f"{sid}.jpg"
        gt_mask_path = data_p / "masks" / f"{sid}.png"
        unet_entry = all_unet_entries.get(sid, {})

        box_override = None
        if perturb:
            clean_box = unet_entry.get("largest_component_bbox")
            if clean_box:
                bw = clean_box[2] - clean_box[0]
                bh = clean_box[3] - clean_box[1]
                pad_x = int(0.25 * bw)
                pad_y = int(0.25 * bh)
                shift_x = max(25, int(0.40 * bw))
                shift_y = max(25, int(0.40 * bh))
                box_override = [
                    max(0, clean_box[0] - pad_x + shift_x),
                    max(0, clean_box[1] - pad_y + shift_y),
                    min(476, clean_box[2] + pad_x + shift_x),
                    min(479, clean_box[3] + pad_y + shift_y),
                ]

        print(f"Evaluating [{role}] {sid[:8]} (Class: {lbl})...")
        res = agent.triage_image(
            image_path=img_path,
            unet_pred_entry=unet_entry,
            data_dir=data_p,
            box_override=box_override,
        )

        # Ground-truth scoring
        iou_gt_init = 0.0
        iou_gt_final = 0.0
        if gt_mask_path.exists():
            gt_raw = np.array(Image.open(gt_mask_path).convert("L"))
            gt_bin = preprocess_mask(gt_raw, threshold=128, zero_border=True)
            if res.initial_sam_mask is not None:
                iou_gt_init = compute_iou(res.initial_sam_mask, gt_bin)
            if res.final_sam_mask is not None:
                iou_gt_final = compute_iou(res.final_sam_mask, gt_bin)

        d = res.to_dict()
        d["true_label"] = lbl
        d["role"] = role
        d["perturbed"] = perturb
        d["iou_gt_initial"] = float(iou_gt_init)
        d["iou_gt_final"] = float(iou_gt_final)
        d["delta_iou_gt"] = float(iou_gt_final - iou_gt_init)
        results_list.append(d)

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS SUMMARY — MODE: {mode.upper()}")
    print("=" * 80)
    for r in results_list:
        knn_str = ", ".join([f"{n.get('class_name', '')}({n.get('distance', 0.0):.2f})" for n in r.get("retrieved_neighbors", [])[:3]])
        print(
            f"{r['image_id'][:8]} | {r['role']:18s} | {r['decision']:18s} | "
            f"IoU_GT: {r['iou_gt_initial']:.4f} -> {r['iou_gt_final']:.4f} ({r['delta_iou_gt']:+.4f}) | "
            f"Strategy: {r['refinement_strategy']:26s} | Top-3: {knn_str}"
        )

    summary: Dict[str, Any] = {
        "mode": mode,
        "triage_model": triage_model,
        "n_samples": len(results_list),
        "results": results_list,
    }

    if mode == "dev-calibration":
        rare_items = [r for r in results_list if r["role"] == "rare_dev"]
        nom_items = [r for r in results_list if r["role"] == "nominal_val"]

        # Agent metrics
        tp = sum(1 for r in rare_items if r["decision"] == "DEFER_RARE_REVIEW")
        fn = len(rare_items) - tp
        fp = sum(1 for r in nom_items if r["decision"] == "DEFER_RARE_REVIEW")
        tn = len(nom_items) - fp

        rare_recall = tp / max(1, len(rare_items))
        nom_fp_rate = fp / max(1, len(nom_items))
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * rare_recall / max(1e-6, precision + rare_recall)

        # Baseline metrics
        b1_tp = sum(1 for r in rare_items if r["baseline1"]["decision"] == "DEFER_RARE_REVIEW")
        b1_fp = sum(1 for r in nom_items if r["baseline1"]["decision"] == "DEFER_RARE_REVIEW")
        b1_rec = b1_tp / max(1, len(rare_items))
        b1_fpr = b1_fp / max(1, len(nom_items))
        b1_p = b1_tp / max(1, b1_tp + b1_fp)
        b1_f1 = 2 * b1_p * b1_rec / max(1e-6, b1_p + b1_rec)

        b2_tp = sum(1 for r in rare_items if r["baseline2"]["decision"] == "DEFER_RARE_REVIEW")
        b2_fp = sum(1 for r in nom_items if r["baseline2"]["decision"] == "DEFER_RARE_REVIEW")
        b2_rec = b2_tp / max(1, len(rare_items))
        b2_fpr = b2_fp / max(1, len(nom_items))
        b2_p = b2_tp / max(1, b2_tp + b2_fp)
        b2_f1 = 2 * b2_p * b2_rec / max(1e-6, b2_p + b2_rec)

        b3_tp = sum(1 for r in rare_items if r["baseline3"]["decision"] == "DEFER_RARE_REVIEW")
        b3_fp = sum(1 for r in nom_items if r["baseline3"]["decision"] == "DEFER_RARE_REVIEW")
        b3_rec = b3_tp / max(1, len(rare_items))
        b3_fpr = b3_fp / max(1, len(nom_items))
        b3_p = b3_tp / max(1, b3_tp + b3_fp)
        b3_f1 = 2 * b3_p * b3_rec / max(1e-6, b3_p + b3_rec)

        refined_items = [r for r in results_list if r["refinement_triggered"] and r["refinement_accepted"]]
        deltas = [r["delta_iou_gt"] for r in refined_items]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        median_delta = float(np.median(deltas)) if deltas else 0.0

        print("\n" + "=" * 80)
        print("DEV CALIBRATION METRICS (34 SAMPLES: 17 rare_dev + 17 nominal_val)")
        print("=" * 80)
        print(f"Agent Rare Recall:              {rare_recall*100:.1f}% ({tp}/{len(rare_items)})")
        print(f"Agent Nominal False-Flag Rate:  {nom_fp_rate*100:.1f}% ({fp}/{len(nom_items)})")
        print(f"Agent Precision / F1:           P={precision*100:.1f}%, F1={f1:.3f}")
        print(f"Refinement Triggered & Accepted: {len(refined_items)} samples")
        print(f"Mean Delta IoU_GT (Refined):    {mean_delta:+.4f}")
        print(f"Median Delta IoU_GT (Refined):  {median_delta:+.4f}")
        print("-" * 80)
        print(f"Baseline 1 (U-Net Max Conf):  Rare Recall={b1_rec*100:.1f}%, Nom FP={b1_fpr*100:.1f}%, F1={b1_f1:.3f}")
        print(f"Baseline 2 (Discordance):     Rare Recall={b2_rec*100:.1f}%, Nom FP={b2_fpr*100:.1f}%, F1={b2_f1:.3f}")
        print(f"Baseline 3 (k-NN / Logistic): Rare Recall={b3_rec*100:.1f}%, Nom FP={b3_fpr*100:.1f}%, F1={b3_f1:.3f}")
        print("=" * 80)

        summary["calibration_metrics"] = {
            "rare_recall": rare_recall,
            "nominal_false_flag_rate": nom_fp_rate,
            "precision": precision,
            "f1": f1,
            "mean_delta_iou_gt_refined": mean_delta,
            "median_delta_iou_gt_refined": median_delta,
            "baselines": {
                "b1": {"recall": b1_rec, "fp_rate": b1_fpr, "f1": b1_f1},
                "b2": {"recall": b2_rec, "fp_rate": b2_fpr, "f1": b2_f1},
                "b3": {"recall": b3_rec, "fp_rate": b3_fpr, "f1": b3_f1},
            },
        }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="WaferGuard Triage Agent Evaluation Runner.")
    parser.add_argument("--mode", type=str, default="dev-dry-run", choices=["sharpened-10", "official-10", "dev-dry-run", "dev-calibration", "full"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--triage-model", "--model",
        dest="triage_model",
        type=str,
        default="muse-spark-1.3",
        help="Triage model: 'muse-spark-1.3' (Meta API) or 'meta-models/muse-spark-open-1026' (Baseten/Modal)",
    )
    parser.add_argument("--api-base-url", type=str, default=None, help="Optional custom API base URL")
    parser.add_argument("--data-dir", type=str, default="data/carinthia_s")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    run_evaluation(
        mode=args.mode,
        output_path=out_p,
        triage_model=args.triage_model,
        api_base_url=args.api_base_url,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
