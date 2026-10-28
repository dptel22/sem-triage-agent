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

    try:
        api_key = resolve_api_key(provider="baseten" if is_open_weights else "meta")
    except ValueError:
        api_key = None

    print("=" * 80)
    print(f"WAFERGUARD TRIAGE AGENT EVALUATION — MODE: {mode.upper()} — MODEL: {triage_model}")
    print("=" * 80)

    # 1. Load Precomputed Retrieval Bank
    bank_cache = PROJECT_ROOT / "outputs" / "gate2" / "retrieval_bank.json"
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
        api_base_url=api_base_url or "https://api.meta.ai/v1",
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
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    for r in results_list:
        print(
            f"{r['image_id'][:8]} | {r['role']:20s} | {r['decision']:20s} | "
            f"IoU_GT: {r['iou_gt_initial']:.4f} -> {r['iou_gt_final']:.4f} ({r['delta_iou_gt']:+.4f}) | "
            f"Strategy: {r['refinement_strategy']}"
        )

    summary = {
        "mode": mode,
        "n_samples": len(results_list),
        "results": results_list,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="WaferGuard Triage Agent Evaluation Runner.")
    parser.add_argument("--mode", type=str, default="dev-dry-run", choices=["sharpened-10", "official-10", "dev-dry-run", "full"])
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
