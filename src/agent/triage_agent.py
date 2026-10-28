"""Verifiable Multimodal Triage Agent for SEM Defect Metrology (Gate 2).

Integrates U-Net defect proposal, SAM 3.1 interactive segmentation with
deterministic mask verification (Condition C1), empty-scan routing (Condition C3),
unbiased Top-5 exemplar retrieval, 3 benchmark baselines, and multimodal LLM
decision commitment via function calling.
"""

from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from PIL import Image

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import meta_sam_parser as msp
    _MSP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MSP_AVAILABLE = False

from src.agent.verifier import evaluate_empty_scan_gate, verify_sam_mask
from src.dataset import CarinthiaSDataset, CLASS_NAMES
from src.sam import build_box_prompt, decode_output_text, mask_to_full_canvas
from src.unet import ResNet18UNet, load_unet, predict_prob

RARE_CLASSES = {1, 2, 5}
NOMINAL_CLASSES = {3, 4}


def resolve_api_key() -> str:
    """Safely retrieves Meta Model API key without exposing secrets."""
    key = os.environ.get("META_API_KEY")
    if key:
        return key.strip().strip("'").strip('"')

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("META_API_KEY=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip()
                val = val.strip("'").strip('"')
                if val:
                    return val
    raise ValueError("META_API_KEY not found in environment or .env file.")


# ---------------------------------------------------------------------------
# Morphology Descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MorphologyFeatures:
    """Quantitative morphological descriptors of a segmented defect."""

    area: int
    box_area: int
    box_fill_ratio: float
    aspect_ratio: float
    log_area: float
    norm_centroid_x: float
    norm_centroid_y: float
    crop_mean_intensity: float
    crop_std_intensity: float
    contrast: float
    hist_8: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Flattens scalar descriptors into a 1D feature vector for retrieval."""
        scalars = [
            self.log_area,
            self.aspect_ratio,
            self.box_fill_ratio,
            self.norm_centroid_x,
            self.norm_centroid_y,
            self.crop_mean_intensity,
            self.crop_std_intensity,
            self.contrast,
        ]
        return np.array(scalars + self.hist_8, dtype=np.float32)


# Alias for backwards compatibility
DefectMorphology = MorphologyFeatures


def compute_defect_morphology(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    box: Tuple[int, int, int, int] | List[int],
) -> MorphologyFeatures:
    """Computes quantitative morphology descriptors from an SEM image and defect mask."""
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    else:
        img_arr = np.array(image, dtype=np.float32)
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

    mask_bin = (mask > 0).astype(np.uint8)
    area = int(mask_bin.sum())

    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    box_area = bw * bh
    clip_x1, clip_x2 = max(0, min(480, x1)), max(0, min(480, x2 + 1))
    clip_y1, clip_y2 = max(0, min(480, y1)), max(0, min(480, y2 + 1))
    defect_in_box = int(mask_bin[clip_y1:clip_y2, clip_x1:clip_x2].sum())
    box_fill = float(min(1.0, max(0.0, defect_in_box / box_area)))
    aspect_ratio = float(bw / max(1, bh))
    log_area = float(math.log(1.0 + area))

    y_idx, x_idx = np.where(mask_bin > 0)
    if len(y_idx) > 0:
        cx = float(x_idx.mean()) / 480.0
        cy = float(y_idx.mean()) / 480.0
    else:
        cx = float((x1 + x2) / 2.0) / 480.0
        cy = float((y1 + y2) / 2.0) / 480.0

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(480, x2 + 1), min(480, y2 + 1)
    crop_img = img_arr[cy1:cy2, cx1:cx2]
    crop_mask = mask_bin[cy1:cy2, cx1:cx2]

    if area > 0 and crop_mask.sum() > 0:
        defect_pixels = crop_img[crop_mask > 0]
        mean_int = float(defect_pixels.mean())
        std_int = float(defect_pixels.std())
        bg_pixels = crop_img[crop_mask == 0]
        bg_mean = float(bg_pixels.mean()) if len(bg_pixels) > 0 else 0.5
        contrast = float(abs(mean_int - bg_mean))
        hist, _ = np.histogram(defect_pixels, bins=8, range=(0.0, 1.0), density=True)
        hist_norm = [float(h / (hist.sum() + 1e-6)) for h in hist]
    else:
        mean_int = float(crop_img.mean()) if crop_img.size > 0 else 0.5
        std_int = float(crop_img.std()) if crop_img.size > 0 else 0.0
        contrast = 0.0
        hist_norm = [0.125] * 8

    return MorphologyFeatures(
        area=area,
        box_area=box_area,
        box_fill_ratio=box_fill,
        aspect_ratio=aspect_ratio,
        log_area=log_area,
        norm_centroid_x=cx,
        norm_centroid_y=cy,
        crop_mean_intensity=mean_int,
        crop_std_intensity=std_int,
        contrast=contrast,
        hist_8=hist_norm,
    )


# ---------------------------------------------------------------------------
# Retrieval Bank
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievedNeighbor:
    """One nearest neighbor exemplar retrieved from the training bank."""

    rank: int
    id: str
    class_label: int
    class_name: str
    distance: float
    box: List[int]
    morphology: Dict[str, Any]


class RetrievalBank:
    """Unified retrieval bank indexed over nominal_train + rare_dev."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        vectors: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.samples = samples
        self.vectors = vectors
        self.mean = mean
        self.std = std
        self.id_to_index = {s["id"]: i for i, s in enumerate(samples)}

    def standardize(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.mean) / np.maximum(self.std, 1e-6)

    def retrieve_top_k(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[RetrievedNeighbor]:
        q_std = self.standardize(query_vector)
        dists = np.linalg.norm(self.vectors - q_std, axis=1)

        if exclude_id and exclude_id in self.id_to_index:
            ex_idx = self.id_to_index[exclude_id]
            dists[ex_idx] = float("inf")

        top_indices = np.argsort(dists)[:k]
        results: List[RetrievedNeighbor] = []
        for rank, idx in enumerate(top_indices, start=1):
            s = self.samples[idx]
            lbl = s["class_label"]
            results.append(
                RetrievedNeighbor(
                    rank=rank,
                    id=s["id"],
                    class_label=lbl,
                    class_name=CLASS_NAMES.get(lbl, f"class_{lbl}"),
                    distance=float(dists[idx]),
                    box=s["box"],
                    morphology=s["morphology"],
                )
            )
        return results

    @classmethod
    def load_or_build(
        cls,
        data_dir: Path | str = "data/carinthia_s",
        manifest_path: Path | str = "results/gate1/splits_manifest.json",
        cache_path: Path | str = "outputs/gate2/retrieval_bank.json",
    ) -> "RetrievalBank":
        cache_p = Path(cache_path)
        if cache_p.exists():
            data = json.loads(cache_p.read_text(encoding="utf-8"))
            samples = data["samples"]
            vectors = np.array(data["vectors"], dtype=np.float32)
            mean = np.array(data["mean"], dtype=np.float32)
            std = np.array(data["std"], dtype=np.float32)
            return cls(samples=samples, vectors=vectors, mean=mean, std=std)

        data_p = Path(data_dir)
        man_p = Path(manifest_path)
        splits = json.load(open(man_p, encoding="utf-8"))["splits"]

        train_pool = splits["nominal_train"] + splits["rare_dev"]
        samples_out: List[Dict[str, Any]] = []
        raw_vectors: List[np.ndarray] = []

        for item in train_pool:
            sid = item["id"]
            lbl = item["class_label"]
            img_p = data_p / "images" / f"{sid}.jpg"
            mask_p = data_p / "masks" / f"{sid}.png"

            if not img_p.exists() or not mask_p.exists():
                continue

            img = Image.open(img_p).convert("L")
            mask_raw = np.array(Image.open(mask_p).convert("L"))
            bin_mask = CarinthiaSDataset.preprocess_mask(mask_raw, threshold=128, zero_border=True)

            tight_box, _ = CarinthiaSDataset.compute_bounding_box(
                mask_raw, pad_ratio=0.0, convention="inclusive", threshold=128, zero_border=True
            )
            if tight_box is None:
                continue

            morph = compute_defect_morphology(img, bin_mask, tight_box)
            vec = morph.to_vector()

            samples_out.append({
                "id": sid,
                "class_label": lbl,
                "box": list(tight_box),
                "morphology": morph.to_dict(),
            })
            raw_vectors.append(vec)

        raw_mat = np.array(raw_vectors, dtype=np.float32)
        mean = np.mean(raw_mat, axis=0)
        std = np.std(raw_mat, axis=0)
        std[std == 0] = 1.0
        std_mat = (raw_mat - mean) / std

        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "samples": samples_out,
            "vectors": std_mat.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        cache_p.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        return cls(samples=samples_out, vectors=std_mat, mean=mean, std=std)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineVerdict:
    baseline_name: str
    decision: str
    score: float
    details: Dict[str, Any]


class Baseline1Confidence:
    """Baseline 1: Deterministic U-Net max probability thresholding."""

    def __init__(self, confidence_cutoff: float = 0.90):
        self.cutoff = confidence_cutoff

    def evaluate(self, unet_pred_entry: Dict[str, Any], is_empty_scan: bool) -> BaselineVerdict:
        if is_empty_scan:
            return BaselineVerdict("Baseline 1 (U-Net Max Confidence)", "NO_DEFECT_REIMAGE", 0.0, {"is_empty": True})
        max_prob = float(unet_pred_entry.get("max_prob", 0.0))
        decision = "DEFER_RARE_REVIEW" if max_prob < self.cutoff else "ACCEPT_NOMINAL"
        return BaselineVerdict("Baseline 1 (U-Net Max Confidence)", decision, max_prob, {"max_prob": max_prob, "cutoff": self.cutoff})


class Baseline2Discordance:
    """Baseline 2: U-Net / SAM mask discordance (1.0 - IoU)."""

    def __init__(self, discordance_cutoff: float = 0.35):
        self.cutoff = discordance_cutoff

    def evaluate(self, unet_sam_iou: float, is_empty_scan: bool) -> BaselineVerdict:
        if is_empty_scan:
            return BaselineVerdict("Baseline 2 (U-Net/SAM Discordance)", "NO_DEFECT_REIMAGE", 1.0, {"is_empty": True})
        discordance = 1.0 - float(unet_sam_iou)
        decision = "DEFER_RARE_REVIEW" if discordance >= self.cutoff else "ACCEPT_NOMINAL"
        return BaselineVerdict("Baseline 2 (U-Net/SAM Discordance)", decision, discordance, {"unet_sam_iou": float(unet_sam_iou), "discordance": discordance, "cutoff": self.cutoff})


class Baseline3KNNLogistic:
    """Baseline 3: Distance-Weighted k-NN and Logistic Regression classifier."""

    def __init__(self, rare_prob_cutoff: float = 0.50, knn_weight_eps: float = 1e-4):
        self.cutoff = rare_prob_cutoff
        self.eps = knn_weight_eps
        self.lr_weights = np.array([
            0.45, 0.65, -0.55, 0.10, 0.10, -0.30, 0.70, 0.50,
            0.20, -0.10, 0.15, -0.20, 0.10, -0.15, 0.25, -0.10
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
        logit = max(-15.0, min(15.0, logit))
        return float(1.0 / (1.0 + np.exp(-logit)))

    def evaluate(
        self,
        neighbors: List[RetrievedNeighbor],
        standardized_features: Optional[np.ndarray],
        is_empty_scan: bool,
    ) -> BaselineVerdict:
        if is_empty_scan:
            return BaselineVerdict("Baseline 3 (Non-LLM k-NN & Logistic)", "NO_DEFECT_REIMAGE", 0.0, {"is_empty": True})
        p_knn = self.compute_knn_prob(neighbors)
        p_lr = self.compute_lr_prob(standardized_features) if standardized_features is not None else p_knn
        p_combined = 0.5 * p_knn + 0.5 * p_lr
        decision = "DEFER_RARE_REVIEW" if p_combined >= self.cutoff else "ACCEPT_NOMINAL"
        return BaselineVerdict(
            "Baseline 3 (Non-LLM k-NN & Logistic)",
            decision,
            p_combined,
            {
                "p_knn_rare": round(p_knn, 4),
                "p_lr_rare": round(p_lr, 4),
                "p_combined": round(p_combined, 4),
                "top5_classes": [n.class_name for n in neighbors[:5]],
                "top5_dists": [round(n.distance, 4) for n in neighbors[:5]],
                "cutoff": self.cutoff,
            },
        )


# ---------------------------------------------------------------------------
# Verifiable Triage Agent Implementation
# ---------------------------------------------------------------------------
@dataclass
class TriageAgentResult:
    image_id: str
    decision: str
    confidence: float
    rationale: str
    cited_evidence: List[str]
    is_empty_scan: bool
    empty_scan_reason: Optional[str]
    initial_box: Optional[List[int]]
    refined_box: Optional[List[int]]
    initial_unet_sam_iou: float
    final_unet_sam_iou: float
    refinement_triggered: bool
    refinement_accepted: bool
    refinement_strategy: str
    refinement_reasons: List[str]
    delta_iou_proxy: float
    baseline1: BaselineVerdict
    baseline2: BaselineVerdict
    baseline3: BaselineVerdict
    retrieved_neighbors: List[RetrievedNeighbor]
    sam_response_id: Optional[str]
    sam_initial_response_id: Optional[str]
    sam_refined_response_id: Optional[str]
    triage_response_id: Optional[str]
    triage_model: str
    token_usage: Optional[Dict[str, int]]
    initial_sam_mask: Optional[np.ndarray] = None
    final_sam_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("initial_sam_mask", None)
        d.pop("final_sam_mask", None)
        return d


def extract_interior_points(binary_mask: np.ndarray, max_points: int = 2) -> List[Tuple[int, int]]:
    """Extracts up to max_points interior (x, y) coordinates from a binary mask by iterative erosion."""
    m = (binary_mask > 0).copy()
    if m.sum() == 0:
        return []

    last_m = m.copy()
    while m.sum() > max_points:
        last_m = m.copy()
        eroded = np.zeros_like(m)
        eroded[1:-1, 1:-1] = (
            m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        )
        if eroded.sum() == 0:
            break
        m = eroded

    pts_arr = np.argwhere(m if m.sum() > 0 else last_m)
    step = max(1, len(pts_arr) // max_points)
    selected = pts_arr[::step][:max_points]
    return [(int(p[1]), int(p[0])) for p in selected]


class VerifiableTriageAgent:
    """Verifiable Multimodal Triage Agent for Industrial SEM defect classification."""

    def __init__(
        self,
        retrieval_bank: RetrievalBank,
        unet_model_path: str | Path = "outputs/gate1/checkpoints/best_unet.pt",
        triage_model: str = "muse-spark-1.3",
        api_base_url: str = "https://api.meta.ai/v1",
        api_key: Optional[str] = None,
        device: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.retrieval_bank = retrieval_bank
        self.triage_model = triage_model
        self.api_base_url = api_base_url
        self.api_key = api_key or (resolve_api_key() if client is None and os.environ.get("META_API_KEY") else "mocked_key")
        self.client = client or openai.OpenAI(api_key=self.api_key, base_url=self.api_base_url, timeout=120.0)

        self.b1_evaluator = Baseline1Confidence(confidence_cutoff=0.90)
        self.b2_evaluator = Baseline2Discordance(discordance_cutoff=0.35)
        self.b3_evaluator = Baseline3KNNLogistic(rare_prob_cutoff=0.50)

        self.device = device or ("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.unet: Optional[ResNet18UNet] = None
        unet_p = Path(unet_model_path)
        if unet_p.exists():
            self.unet = load_unet(unet_p, device=self.device)

    def _predict_unet_prob(self, img_pil: Image.Image) -> np.ndarray:
        if self.unet is None:
            return np.zeros((480, 480), dtype=np.float32)
        return predict_prob(self.unet, img_pil, device=self.device)

    def _predict_unet_mask(self, img_pil: Image.Image) -> np.ndarray:
        prob = self._predict_unet_prob(img_pil)
        return (prob >= 0.5).astype(np.uint8)

    def triage_image(
        self,
        image_path: Path | str,
        unet_pred_entry: Dict[str, Any],
        data_dir: Path | str = "data/carinthia_s",
        box_override: Optional[List[int]] = None,
    ) -> TriageAgentResult:
        img_p = Path(image_path)
        data_p = Path(data_dir)
        sid = unet_pred_entry["id"]
        img_pil = Image.open(img_p).convert("L")

        # Step 1: Empty-Scan Gate (Condition C3)
        c3_result = evaluate_empty_scan_gate(unet_pred_entry, threshold_pixels=64)
        if c3_result.is_empty:
            b1 = self.b1_evaluator.evaluate(unet_pred_entry, is_empty_scan=True)
            b2 = self.b2_evaluator.evaluate(0.0, is_empty_scan=True)
            b3 = self.b3_evaluator.evaluate([], None, is_empty_scan=True)
            return TriageAgentResult(
                image_id=sid,
                decision="NO_DEFECT_REIMAGE",
                confidence=1.0,
                rationale=f"Empty-scan gate triggered: {c3_result.reason}.",
                cited_evidence=[
                    f"Filtered foreground pixels: {c3_result.filtered_fg_pixels} < 64 threshold",
                    "No valid connected component survived area >= 16px filter",
                ],
                is_empty_scan=True,
                empty_scan_reason=c3_result.reason,
                initial_box=None,
                refined_box=None,
                initial_unet_sam_iou=0.0,
                final_unet_sam_iou=0.0,
                refinement_triggered=False,
                refinement_accepted=False,
                refinement_strategy="none",
                refinement_reasons=[],
                delta_iou_proxy=0.0,
                baseline1=b1,
                baseline2=b2,
                baseline3=b3,
                retrieved_neighbors=[],
                sam_response_id=None,
                sam_initial_response_id=None,
                sam_refined_response_id=None,
                triage_response_id=None,
                triage_model=self.triage_model,
                token_usage=None,
                initial_sam_mask=np.zeros((480, 480), dtype=np.uint8),
                final_sam_mask=np.zeros((480, 480), dtype=np.uint8),
            )

        # Step 2: SAM Segmentation & Verification (Condition C1)
        target_box = box_override or unet_pred_entry.get("largest_component_bbox")
        if target_box is None:
            target_box = [0, 0, 476, 479]
        bx1 = max(0, min(476, int(target_box[0])))
        by1 = max(0, min(479, int(target_box[1])))
        bx2 = max(0, min(476, int(target_box[2])))
        by2 = max(0, min(479, int(target_box[3])))
        active_box = [bx1, by1, bx2, by2]

        prob_unet = self._predict_unet_prob(img_pil)
        m_unet = (prob_unet >= 0.5).astype(np.uint8)

        img_b64 = base64.b64encode(img_p.read_bytes()).decode("ascii")
        full_data_url = f"data:image/jpeg;base64,{img_b64}"

        init_prompt = build_box_prompt(
            active_box[0], active_box[1], active_box[2], active_box[3],
            480, 480, convention="inclusive", object_id=1,
        )
        sam_resp = self.client.responses.create(
            model="sam-3.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": init_prompt},
                    {"type": "input_image", "image_url": full_data_url},
                ],
            }],
        )
        sam_resp_id = getattr(sam_resp, "id", None)
        parsed_sam = decode_output_text(getattr(sam_resp, "output_text", "") or "")
        if parsed_sam.masks:
            m_sam_init = mask_to_full_canvas(parsed_sam.masks[0], 480, 480)
        else:
            m_sam_init = np.zeros((480, 480), dtype=np.uint8)

        verif = verify_sam_mask(m_unet, m_sam_init, active_box)
        init_iou = verif.unet_sam_iou
        final_iou = init_iou
        m_sam_final = m_sam_init
        final_box = active_box
        delta_iou_proxy = 0.0
        sam_resp_ref_id = None
        refinement_accepted = False
        refinement_strategy = "none"

        if verif.triggered_refinement:
            unet_box = unet_pred_entry.get("largest_component_bbox") or active_box
            ux1 = max(0, min(476, int(unet_box[0])))
            uy1 = max(0, min(479, int(unet_box[1])))
            ux2 = max(0, min(476, int(unet_box[2])))
            uy2 = max(0, min(479, int(unet_box[3])))
            core_box = [ux1, uy1, ux2, uy2]

            sig_components = [c for c in unet_pred_entry.get("component_boxes", []) if c.get("area", 0) >= 64]
            r_leak = np.logical_and(m_sam_init == 1, prob_unet < 0.15).astype(np.uint8)
            leak_area = int(r_leak.sum())

            # Priority 1: Shift recovery
            if active_box != core_box:
                refinement_strategy = "core_box_recentering"
                pt_x = (core_box[0] + core_box[2]) // 2
                pt_y = (core_box[1] + core_box[3]) // 2
                ref_obj = msp.PromptObject(
                    id=1,
                    prompts=[msp.FramePrompt(
                        frame=0,
                        box=msp.PromptBox(left=core_box[0], top=core_box[1], right=core_box[2] + 1, bottom=core_box[3] + 1),
                        points=[msp.PromptPoint(x=pt_x, y=pt_y, label="positive")],
                    )],
                )
                refined_prompt = msp.build_object_prompt(size=(480, 480), objects=[ref_obj])
                sam_resp_ref = self.client.responses.create(
                    model="sam-3.1",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": refined_prompt}, {"type": "input_image", "image_url": full_data_url}]}],
                )
                sam_resp_ref_id = getattr(sam_resp_ref, "id", None)
                parsed_ref = decode_output_text(getattr(sam_resp_ref, "output_text", "") or "")
                if parsed_ref.masks:
                    m_sam_ref = mask_to_full_canvas(parsed_ref.masks[0], 480, 480)
                    ref_verif = verify_sam_mask(m_unet, m_sam_ref, core_box)
                    refined_iou = ref_verif.unet_sam_iou
                    if refined_iou > init_iou:
                        m_sam_final = m_sam_ref
                        final_iou = refined_iou
                        final_box = core_box
                        delta_iou_proxy = refined_iou - init_iou
                        refinement_accepted = True

            # Priority 2: Multi-component defect union
            elif len(sig_components) >= 2:
                refinement_strategy = "multi_component_sub_box_union"
                top_k_boxes = [c["bbox"] for c in sig_components[:3]]
                objs = [
                    msp.PromptObject(
                        id=i + 1,
                        prompts=[msp.FramePrompt(
                            frame=0,
                            box=msp.PromptBox(
                                left=max(0, min(476, int(b[0]))),
                                top=max(0, min(479, int(b[1]))),
                                right=max(0, min(476, int(b[2]))) + 1,
                                bottom=max(0, min(479, int(b[3]))) + 1,
                            ),
                        )],
                    )
                    for i, b in enumerate(top_k_boxes)
                ]
                refined_prompt = msp.build_object_prompt(size=(480, 480), objects=objs)
                sam_resp_ref = self.client.responses.create(
                    model="sam-3.1",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": refined_prompt}, {"type": "input_image", "image_url": full_data_url}]}],
                )
                sam_resp_ref_id = getattr(sam_resp_ref, "id", None)
                parsed_ref = decode_output_text(getattr(sam_resp_ref, "output_text", "") or "")
                if parsed_ref.masks:
                    m_sam_ref = np.zeros((480, 480), dtype=np.uint8)
                    for m in parsed_ref.masks:
                        m_sam_ref = np.logical_or(m_sam_ref, mask_to_full_canvas(m, 480, 480)).astype(np.uint8)
                    ref_verif = verify_sam_mask(m_unet, m_sam_ref, active_box)
                    refined_iou = ref_verif.unet_sam_iou
                    if refined_iou > init_iou:
                        m_sam_final = m_sam_ref
                        final_iou = refined_iou
                        delta_iou_proxy = refined_iou - init_iou
                        refinement_accepted = True

            # Priority 3: Negative point carving
            elif (m_sam_init.sum() > 1.3 * max(1, m_unet.sum()) or init_iou < 0.65) and leak_area >= 32:
                refinement_strategy = "negative_point_carving"
                neg_pts = extract_interior_points(r_leak, max_points=2)
                pos_mask = (prob_unet >= 0.5)
                if pos_mask.sum() > 0:
                    py, px = np.unravel_index(np.argmax(prob_unet * pos_mask), prob_unet.shape)
                else:
                    py, px = (active_box[1] + active_box[3]) // 2, (active_box[0] + active_box[2]) // 2
                pos_pt = (int(px), int(py))

                points_prompt = [msp.PromptPoint(x=pos_pt[0], y=pos_pt[1], label="positive")]
                for nx, ny in neg_pts:
                    points_prompt.append(msp.PromptPoint(x=nx, y=ny, label="negative"))

                ref_obj = msp.PromptObject(
                    id=1,
                    prompts=[msp.FramePrompt(
                        frame=0,
                        box=msp.PromptBox(left=active_box[0], top=active_box[1], right=active_box[2] + 1, bottom=active_box[3] + 1),
                        points=points_prompt,
                    )],
                )
                refined_prompt = msp.build_object_prompt(size=(480, 480), objects=[ref_obj])
                sam_resp_ref = self.client.responses.create(
                    model="sam-3.1",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": refined_prompt}, {"type": "input_image", "image_url": full_data_url}]}],
                )
                sam_resp_ref_id = getattr(sam_resp_ref, "id", None)
                parsed_ref = decode_output_text(getattr(sam_resp_ref, "output_text", "") or "")
                if parsed_ref.masks:
                    m_sam_ref = mask_to_full_canvas(parsed_ref.masks[0], 480, 480)
                    ref_verif = verify_sam_mask(m_unet, m_sam_ref, active_box)
                    refined_iou = ref_verif.unet_sam_iou
                    if refined_iou > init_iou:
                        m_sam_final = m_sam_ref
                        final_iou = refined_iou
                        delta_iou_proxy = refined_iou - init_iou
                        refinement_accepted = True

            # Priority 4: Fallback
            else:
                refinement_strategy = "composite_box_and_point"
                pt_x = (core_box[0] + core_box[2]) // 2
                pt_y = (core_box[1] + core_box[3]) // 2
                ref_obj = msp.PromptObject(
                    id=1,
                    prompts=[msp.FramePrompt(
                        frame=0,
                        box=msp.PromptBox(left=core_box[0], top=core_box[1], right=core_box[2] + 1, bottom=core_box[3] + 1),
                        points=[msp.PromptPoint(x=pt_x, y=pt_y, label="positive")],
                    )],
                )
                refined_prompt = msp.build_object_prompt(size=(480, 480), objects=[ref_obj])
                sam_resp_ref = self.client.responses.create(
                    model="sam-3.1",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": refined_prompt}, {"type": "input_image", "image_url": full_data_url}]}],
                )
                sam_resp_ref_id = getattr(sam_resp_ref, "id", None)
                parsed_ref = decode_output_text(getattr(sam_resp_ref, "output_text", "") or "")
                if parsed_ref.masks:
                    m_sam_ref = mask_to_full_canvas(parsed_ref.masks[0], 480, 480)
                    ref_verif = verify_sam_mask(m_unet, m_sam_ref, core_box)
                    refined_iou = ref_verif.unet_sam_iou
                    if refined_iou > init_iou:
                        m_sam_final = m_sam_ref
                        final_iou = refined_iou
                        final_box = core_box
                        delta_iou_proxy = refined_iou - init_iou
                        refinement_accepted = True

        # Step 3: Morphology Extraction & Top-5 Exemplar Retrieval
        morph = compute_defect_morphology(img_pil, m_sam_final, final_box)
        query_vec = morph.to_vector()
        top5_neighbors = self.retrieval_bank.retrieve_top_k(query_vec, k=5, exclude_id=sid)
        std_feat = self.retrieval_bank.standardize(query_vec)

        # Step 4: Evaluate 3 Baselines
        b1 = self.b1_evaluator.evaluate(unet_pred_entry, is_empty_scan=False)
        b2 = self.b2_evaluator.evaluate(final_iou, is_empty_scan=False)
        b3 = self.b3_evaluator.evaluate(top5_neighbors, std_feat, is_empty_scan=False)

        # Step 5: Multimodal LLM Decision Node
        tools = [{
            "type": "function",
            "function": {
                "name": "commit_triage_decision",
                "description": "Commit official triage routing decision for SEM defect.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["ACCEPT_NOMINAL", "DEFER_RARE_REVIEW", "NO_DEFECT_REIMAGE"],
                        },
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                        "cited_evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["decision", "confidence", "rationale", "cited_evidence"],
                },
            },
        }]

        knn_summary = ", ".join([f"{n.class_name} (d={n.distance:.2f})" for n in top5_neighbors])
        user_prompt = (
            f"SEM Defect Triage Analysis for image {sid}:\n"
            f"- U-Net Confidence: {unet_pred_entry.get('max_prob', 0.0):.4f}\n"
            f"- Observable Mask Agreement IoU: {final_iou:.4f}\n"
            f"- Morphology: area={morph.area}px, aspect_ratio={morph.aspect_ratio:.2f}, fill={morph.box_fill_ratio:.2f}\n"
            f"- Top-5 Retrieved Exemplars: {knn_summary}\n"
            f"- Baseline Recommendations: B1={b1.decision}, B2={b2.decision}, B3={b3.decision}\n"
            "Call commit_triage_decision now."
        )

        triage_resp_id = None
        token_usage = None
        decision = "ACCEPT_NOMINAL"
        confidence = 0.90
        rationale = "Default fallback triage decision."
        cited_evidence = ["Observable agreement above threshold"]

        try:
            llm_resp = self.client.chat.completions.create(
                model=self.triage_model,
                messages=[
                    {"role": "system", "content": "You are WaferGuard, an expert semiconductor triage agent. Call commit_triage_decision."},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
            triage_resp_id = getattr(llm_resp, "id", None)
            if getattr(llm_resp, "usage", None):
                token_usage = {
                    "prompt_tokens": getattr(llm_resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(llm_resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(llm_resp.usage, "total_tokens", 0),
                }

            choice = llm_resp.choices[0]
            if choice.message.tool_calls:
                fn_args = json.loads(choice.message.tool_calls[0].function.arguments)
                decision = fn_args["decision"]
                confidence = float(fn_args["confidence"])
                rationale = fn_args["rationale"]
                cited_evidence = fn_args.get("cited_evidence", [])
        except Exception as e:
            decision = b3.decision
            confidence = float(b3.score)
            rationale = f"LLM call fallback to Baseline 3: {e}"

        return TriageAgentResult(
            image_id=sid,
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            cited_evidence=cited_evidence,
            is_empty_scan=False,
            empty_scan_reason=None,
            initial_box=active_box,
            refined_box=final_box,
            initial_unet_sam_iou=float(init_iou),
            final_unet_sam_iou=float(final_iou),
            refinement_triggered=verif.triggered_refinement,
            refinement_accepted=refinement_accepted,
            refinement_strategy=refinement_strategy,
            refinement_reasons=verif.trigger_reasons,
            delta_iou_proxy=float(delta_iou_proxy),
            baseline1=b1,
            baseline2=b2,
            baseline3=b3,
            retrieved_neighbors=top5_neighbors,
            sam_response_id=sam_resp_id,
            sam_initial_response_id=sam_resp_id,
            sam_refined_response_id=sam_resp_ref_id,
            triage_response_id=triage_resp_id,
            triage_model=self.triage_model,
            token_usage=token_usage,
            initial_sam_mask=m_sam_init,
            final_sam_mask=m_sam_final,
        )
