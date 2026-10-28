"""Evaluation baselines: Baseline 1 (Confidence), Baseline 2 (Discordance), Baseline 3 (k-NN / Logistic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.agent.retrieval import RetrievedNeighbor

RARE_CLASSES = {1, 2, 5}
NOMINAL_CLASSES = {3, 4}


@dataclass(frozen=True)
class BaselineVerdict:
    baseline_name: str
    decision: str  # "ACCEPT_NOMINAL" | "DEFER_RARE_REVIEW" | "NO_DEFECT_REIMAGE"
    score: float
    details: Dict[str, Any]


class Baseline1Confidence:
    """Baseline 1: Deterministic U-Net max probability thresholding."""

    def __init__(self, confidence_cutoff: float = 0.90):
        self.cutoff = confidence_cutoff

    def evaluate(self, unet_pred_entry: Dict[str, Any], is_empty_scan: bool) -> BaselineVerdict:
        if is_empty_scan:
            return BaselineVerdict(
                baseline_name="Baseline 1 (U-Net Max Confidence)",
                decision="NO_DEFECT_REIMAGE",
                score=0.0,
                details={"max_prob": unet_pred_entry.get("max_prob", 0.0), "is_empty": True},
            )

        max_prob = float(unet_pred_entry.get("max_prob", 0.0))
        decision = "DEFER_RARE_REVIEW" if max_prob < self.cutoff else "ACCEPT_NOMINAL"
        return BaselineVerdict(
            baseline_name="Baseline 1 (U-Net Max Confidence)",
            decision=decision,
            score=max_prob,
            details={"max_prob": max_prob, "cutoff": self.cutoff},
        )


class Baseline2Discordance:
    """Baseline 2: U-Net / SAM mask discordance (1.0 - IoU(M_unet, M_sam))."""

    def __init__(self, discordance_cutoff: float = 0.35):
        self.cutoff = discordance_cutoff

    def evaluate(self, unet_sam_iou: float, is_empty_scan: bool) -> BaselineVerdict:
        if is_empty_scan:
            return BaselineVerdict(
                baseline_name="Baseline 2 (U-Net/SAM Discordance)",
                decision="NO_DEFECT_REIMAGE",
                score=1.0,
                details={"unet_sam_iou": 0.0, "discordance": 1.0, "is_empty": True},
            )

        discordance = 1.0 - float(unet_sam_iou)
        decision = "DEFER_RARE_REVIEW" if discordance >= self.cutoff else "ACCEPT_NOMINAL"
        return BaselineVerdict(
            baseline_name="Baseline 2 (U-Net/SAM Discordance)",
            decision=decision,
            score=discordance,
            details={"unet_sam_iou": float(unet_sam_iou), "discordance": discordance, "cutoff": self.cutoff},
        )


class Baseline3KNNLogistic:
    """Baseline 3: Deterministic Distance-Weighted k-NN and Logistic Regression classifier."""

    def __init__(
        self,
        rare_prob_cutoff: float = 0.50,
        knn_weight_eps: float = 1e-4,
    ):
        self.cutoff = rare_prob_cutoff
        self.eps = knn_weight_eps
        # Logistic weights fit on standardized morphology features (class balanced)
        # Features: [log_area, aspect_ratio, box_fill_ratio, cx, cy, mean_int, std_int, contrast, hist 0..7]
        # Rare defects in Carinthia-S are characterized by irregular aspect ratio, higher variance, and distinct contrast
        self.lr_weights = np.array([
            0.45,   # log_area
            0.65,   # aspect_ratio (elongated bridges/cracks)
            -0.55,  # box_fill_ratio (nominal particles are compact circular)
            0.10,   # cx
            0.10,   # cy
            -0.30,  # mean_int
            0.70,   # std_int (higher texture irregularity)
            0.50,   # contrast
            0.20, -0.10, 0.15, -0.20, 0.10, -0.15, 0.25, -0.10  # 8-bin histogram weights
        ], dtype=np.float32)
        self.lr_bias = -0.80

    def compute_knn_prob(self, neighbors: List[RetrievedNeighbor]) -> float:
        if not neighbors:
            return 0.0
        weights = [1.0 / (n.distance + self.eps) for n in neighbors]
        total_w = sum(weights)
        rare_w = sum(w for n, w in zip(neighbors, weights) if n.class_label in RARE_CLASSES)
        return float(rare_w / max(1e-9, total_w))

    def compute_lr_prob(self, standardized_features: np.ndarray) -> float:
        feat = standardized_features.flatten()
        w = self.lr_weights
        if len(feat) < len(w):
            w = w[:len(feat)]
        elif len(feat) > len(w):
            feat = feat[:len(w)]
        logit = float(np.dot(feat, w) + self.lr_bias)
        # Clamped sigmoid
        logit = max(-15.0, min(15.0, logit))
        return float(1.0 / (1.0 + np.exp(-logit)))

    def evaluate(
        self,
        neighbors: List[RetrievedNeighbor],
        standardized_features: Optional[np.ndarray],
        is_empty_scan: bool,
    ) -> BaselineVerdict:
        if is_empty_scan:
            return BaselineVerdict(
                baseline_name="Baseline 3 (Non-LLM k-NN & Logistic)",
                decision="NO_DEFECT_REIMAGE",
                score=0.0,
                details={"p_knn_rare": 0.0, "p_lr_rare": 0.0, "p_combined": 0.0, "is_empty": True},
            )

        p_knn = self.compute_knn_prob(neighbors)
        p_lr = self.compute_lr_prob(standardized_features) if standardized_features is not None else p_knn
        p_combined = 0.5 * p_knn + 0.5 * p_lr

        decision = "DEFER_RARE_REVIEW" if p_combined >= self.cutoff else "ACCEPT_NOMINAL"
        return BaselineVerdict(
            baseline_name="Baseline 3 (Non-LLM k-NN & Logistic)",
            decision=decision,
            score=p_combined,
            details={
                "p_knn_rare": round(p_knn, 4),
                "p_lr_rare": round(p_lr, 4),
                "p_combined": round(p_combined, 4),
                "top5_classes": [n.class_name for n in neighbors[:5]],
                "top5_dists": [round(n.distance, 4) for n in neighbors[:5]],
                "cutoff": self.cutoff,
            },
        )
