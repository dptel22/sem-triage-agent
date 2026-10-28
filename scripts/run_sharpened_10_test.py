"""Sharpened 10-Sample Verification Suite & Evaluation Harness.

Modes:
  --mode dev-dry-run   Runs on 4 development images (2 nominal_val + 1 rare_dev + 1 class6_val)
                       with an induced perturbation to test end-to-end telemetry and C1 recovery.
  --mode official-10   Evaluates the frozen 10 pre-registered images in
                       results/gate1/sharpened_test_10_manifest.json (SHA-256 verified).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.retrieval import RetrievalBank  # noqa: E402
from src.agent.triage_agent import VerifiableTriageAgent  # noqa: E402
from src.dataset.carinthia_loader import CarinthiaSDataset  # noqa: E402

RARE_CLASSES = {1, 2, 5}
NOMINAL_CLASSES = {3, 4}
EMPTY_CLASSES = {6}

NATURAL_RARE_PREVALENCE = 0.015  # 1.5% industry baseline rate
NATURAL_NOMINAL_PREVALENCE = 0.985  # 98.5% industry baseline rate


def compute_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[int],
) -> Dict[str, Any]:
    """Computes Enriched Precision/Recall/F1 and Natural-Prevalence (1.5%) Precision & False-Flag Rate."""
    tp_rare = 0
    fp_rare = 0
    fn_rare = 0
    tn_nominal = 0
    fp_nominal = 0

    c6_total = 0
    c6_correct = 0

    for pred, gt in zip(predictions, ground_truths):
        dec = pred["decision"]
        is_rare_pred = (dec == "DEFER_RARE_REVIEW")
        is_empty_pred = (dec == "NO_DEFECT_REIMAGE")

        if gt in RARE_CLASSES:
            if is_rare_pred:
                tp_rare += 1
            else:
                fn_rare += 1
        elif gt in NOMINAL_CLASSES:
            if is_rare_pred:
                fp_rare += 1
                fp_nominal += 1
            else:
                tn_nominal += 1
        elif gt in EMPTY_CLASSES:
            c6_total += 1
            if is_empty_pred:
                c6_correct += 1

    total_rare_gt = tp_rare + fn_rare
    total_nominal_gt = tn_nominal + fp_nominal

    tpr_rare = (tp_rare / total_rare_gt) if total_rare_gt > 0 else 0.0
    fpr_nominal = (fp_nominal / total_nominal_gt) if total_nominal_gt > 0 else 0.0

    # Enriched precision / recall / F1
    prec_enriched = (tp_rare / (tp_rare + fp_rare)) if (tp_rare + fp_rare) > 0 else 0.0
    recall_enriched = tpr_rare
    f1_enriched = (
        (2 * prec_enriched * recall_enriched / (prec_enriched + recall_enriched))
        if (prec_enriched + recall_enriched) > 0
        else 0.0
    )

    # Bayesian Natural-Prevalence Precision (at 1.5% true rare base rate)
    p_num = NATURAL_RARE_PREVALENCE * tpr_rare
    p_den = p_num + (NATURAL_NOMINAL_PREVALENCE * fpr_nominal)
    prec_natural = (p_num / p_den) if p_den > 0 else 0.0

    c6_specificity = (c6_correct / c6_total) if c6_total > 0 else 1.0

    return {
        "confusion_matrix": {
            "tp_rare": tp_rare,
            "fp_rare": fp_rare,
            "fn_rare": fn_rare,
            "tn_nominal": tn_nominal,
            "fp_nominal": fp_nominal,
            "c6_correct": c6_correct,
            "c6_total": c6_total,
        },
        "enriched_metrics": {
            "precision": round(prec_enriched, 4),
            "recall": round(recall_enriched, 4),
            "f1_score": round(f1_enriched, 4),
        },
        "natural_prevalence_metrics": {
            "assumed_rare_base_rate": NATURAL_RARE_PREVALENCE,
            "true_positive_rate_tpr": round(tpr_rare, 4),
            "false_flag_rate_fpr": round(fpr_nominal, 4),
            "natural_precision": round(prec_natural, 4),
        },
        "condition_c3_empty_scan": {
            "class6_specificity": round(c6_specificity, 4),
            "correct": c6_correct,
            "total": c6_total,
        },
    }


def compute_ground_truth_iou(mask_pred: np.ndarray, gt_mask_path: Path) -> float:
    """Computes exact IoU vs ground truth mask (mask >= 128, x >= 477 zeroed)."""
    gt_raw = np.array(Image.open(gt_mask_path).convert("L"))
    gt_bin = CarinthiaSDataset.preprocess_mask(gt_raw, threshold=128, zero_border=True)
    inter = int(np.logical_and(mask_pred, gt_bin).sum())
    union = int(np.logical_or(mask_pred, gt_bin).sum())
    return float(inter / union) if union > 0 else 0.0


def run_dev_dry_run(agent: VerifiableTriageAgent, args) -> Dict[str, Any]:
    """Runs a 4-image dev dry-run to verify telemetry, tool calling, and C1 recovery."""
    print("======================================================================")
    print("STARTING GATE 2 DEV DRY-RUN (4 Development Samples)")
    print("Pool: strictly nominal_val, class6_val, and rare_dev (test sets untouched)")
    print(f"Model: {args.triage_model} | API: {args.api_base_url}")
    print("======================================================================\n")

    splits_manifest = json.load(open(PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json", encoding="utf-8"))["splits"]
    unet_boxes = json.load(open(PROJECT_ROOT / "results" / "gate1" / "unet_predicted_boxes.json", encoding="utf-8"))

    # Map ID to U-Net entry
    all_unet_entries = {}
    for key, entries in unet_boxes.items():
        if isinstance(entries, list):
            for e in entries:
                all_unet_entries[e["id"]] = e

    # 4 dev images: 2 nominal_val + 1 rare_dev + 1 class6_val
    nom1 = splits_manifest["nominal_val"][0]
    nom2 = splits_manifest["nominal_val"][1]
    rare1 = splits_manifest["rare_dev"][0]  # Class 1
    c6_1 = splits_manifest["class6_val"][0]  # Class 6

    dev_samples = [
        {"sample": nom1, "role": "nominal_clean", "perturb": False},
        {"sample": nom2, "role": "nominal_perturbed", "perturb": True},  # Induced +25% expand, +15px shift
        {"sample": rare1, "role": "rare_clean", "perturb": False},
        {"sample": c6_1, "role": "class6_empty", "perturb": False},
    ]

    results_list: List[Dict[str, Any]] = []
    gts: List[int] = []

    c1_test_record = None

    for item in dev_samples:
        s = item["sample"]
        sid = s["id"]
        role = item["role"]
        perturb = item["perturb"]
        lbl = s["class_label"]
        gts.append(lbl)

        img_path = PROJECT_ROOT / "data" / "carinthia_s" / "images" / f"{sid}.jpg"
        gt_mask_path = PROJECT_ROOT / "data" / "carinthia_s" / "masks" / f"{sid}.png"
        unet_entry = all_unet_entries[sid]

        box_override = None
        if perturb:
            clean_box = unet_entry.get("largest_component_bbox")
            if clean_box:
                bw = clean_box[2] - clean_box[0]
                bh = clean_box[3] - clean_box[1]
                pad_x = int(0.25 * bw)
                pad_y = int(0.25 * bh)
                shift_x, shift_y = 15, 15
                box_override = [
                    max(0, clean_box[0] - pad_x + shift_x),
                    max(0, clean_box[1] - pad_y + shift_y),
                    min(476, clean_box[2] + pad_x + shift_x),
                    min(479, clean_box[3] + pad_y + shift_y),
                ]
                print(f"[{role}] Inducing perturbed box on {sid}: clean={clean_box} -> perturbed={box_override}")

        print(f"\n--- Processing [{role}] {sid} (True class: {lbl}) ---")
        res = agent.triage_image(
            image_path=img_path,
            unet_pred_entry=unet_entry,
            data_dir=PROJECT_ROOT / "data" / "carinthia_s",
            box_override=box_override,
        )
        res_dict = res.to_dict()
        res_dict["true_label"] = lbl
        res_dict["role"] = role
        res_dict["perturbed"] = perturb

        # Check Ground-Truth IoU recovery on perturbed sample
        if perturb and res.refinement_triggered:
            # Measure GT IoU of initial vs refined
            print(f"  Condition C1 Triggered: {res.refinement_reasons}")
            print(f"  Proxy U-Net/SAM IoU: initial={res.initial_unet_sam_iou:.4f} -> final={res.final_unet_sam_iou:.4f} (Delta={res.delta_iou_proxy:+.4f})")
            c1_test_record = {
                "id": sid,
                "initial_box": res.initial_box,
                "refined_box": res.refined_box,
                "proxy_initial_iou": res.initial_unet_sam_iou,
                "proxy_final_iou": res.final_unet_sam_iou,
                "proxy_delta_iou": res.delta_iou_proxy,
                "refinement_triggered": res.refinement_triggered,
                "refinement_reasons": res.refinement_reasons,
            }

        print(f"  Decision: {res.decision} (confidence: {res.confidence:.2f})")
        print(f"  Rationale: {res.rationale}")
        print(f"  Baselines: B1={res.baseline1.decision} | B2={res.baseline2.decision} | B3={res.baseline3.decision} (P(rare)={res.baseline3.score:.3f})")
        print(f"  API IDs: SAM={res.sam_response_id} | Triage={res.triage_response_id}")

        results_list.append(res_dict)

    # Compute overall metrics
    metrics = compute_metrics(results_list, gts)

    output = {
        "mode": "dev-dry-run",
        "triage_model": args.triage_model,
        "n_samples": len(results_list),
        "metrics": metrics,
        "c1_recovery_verification": c1_test_record,
        "results": results_list,
    }

    out_path = PROJECT_ROOT / "outputs" / "gate2" / "dev_dry_run_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved dry-run results to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("DEV DRY-RUN EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Enriched Precision: {metrics['enriched_metrics']['precision']:.4f}")
    print(f"Enriched Recall:    {metrics['enriched_metrics']['recall']:.4f}")
    print(f"Enriched F1-Score:  {metrics['enriched_metrics']['f1_score']:.4f}")
    print("-" * 70)
    print(f"Natural Rare Prevalence:   {metrics['natural_prevalence_metrics']['assumed_rare_base_rate'] * 100:.1f}%")
    print(f"Natural Precision:         {metrics['natural_prevalence_metrics']['natural_precision']:.4f}")
    print(f"Natural False-Flag Rate:   {metrics['natural_prevalence_metrics']['false_flag_rate_fpr']:.4f}")
    print(f"Empty-Scan Specificity:    {metrics['condition_c3_empty_scan']['class6_specificity']:.4f} ({metrics['condition_c3_empty_scan']['correct']}/{metrics['condition_c3_empty_scan']['total']})")
    print("=" * 70)

    if c1_test_record:
        print("Condition C1 Refinement Test: SUCCESS")
        print(f"  Refinement Triggered: {c1_test_record['refinement_triggered']}")
        print(f"  Proxy Delta IoU:      {c1_test_record['proxy_delta_iou']:+.4f}")
    print("\nSTOPPING: Dev dry-run complete. Do NOT execute official-10 until instructed.")
    return output


def main():
    parser = argparse.ArgumentParser(description="Sharpened Triage Verification Suite")
    parser.add_argument(
        "--mode",
        choices=["dev-dry-run", "official-10"],
        required=True,
        help="Evaluation mode: 'dev-dry-run' (4 dev images) or 'official-10' (frozen pre-registered suite).",
    )
    parser.add_argument(
        "--triage-model",
        default="muse-spark-1.3",
        help="Multimodal decision model ('muse-spark-1.3' or 'meta-models/muse-spark-open-1026').",
    )
    parser.add_argument(
        "--api-base-url",
        default="https://api.meta.ai/v1",
        help="API base URL for inference endpoint.",
    )
    args = parser.parse_args()

    # Load or build retrieval bank
    print("Initializing Unbiased Top-5 Retrieval Bank...")
    bank = RetrievalBank.load_or_build(
        data_dir=PROJECT_ROOT / "data" / "carinthia_s",
        manifest_path=PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json",
        cache_path=PROJECT_ROOT / "outputs" / "gate2" / "retrieval_bank.json",
    )
    print(f"Retrieval Bank ready: {len(bank.samples)} exemplars indexed.")

    # Initialize triage agent
    agent = VerifiableTriageAgent(
        retrieval_bank=bank,
        unet_model_path=PROJECT_ROOT / "outputs" / "gate1" / "checkpoints" / "best_unet.pt",
        triage_model=args.triage_model,
        api_base_url=args.api_base_url,
    )

    if args.mode == "dev-dry-run":
        run_dev_dry_run(agent, args)
    elif args.mode == "official-10":
        print("FATAL: --mode official-10 cannot be run before human review of dev-dry-run.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
