"""Verifiable Triage Agent for SEM Industrial Defect Inspection.

Implements the zero-leakage blinding contract:
The agent takes ONLY (image_path, unet_pred_entry)—never ground-truth masks,
ground-truth boxes, or ground-truth class labels.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import meta_sam_parser as msp
import numpy as np
import openai
import torch
from PIL import Image, ImageDraw

from src.agent.baselines import (
    Baseline1Confidence,
    Baseline2Discordance,
    Baseline3KNNLogistic,
    BaselineVerdict,
)
from src.agent.morphology import compute_defect_morphology
from src.agent.retrieval import RetrievalBank, RetrievedNeighbor
from src.agent.verifier import evaluate_empty_scan_gate, verify_sam_mask
from src.api import sam_decoder
from scripts.train_unet_baseline import ResNet18UNet

COMMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "commit_triage_decision",
        "description": "Commit the final visual triage decision for an industrial SEM part image.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["ACCEPT_NOMINAL", "DEFER_RARE_REVIEW", "NO_DEFECT_REIMAGE"],
                    "description": (
                        "Final triage verdict: ACCEPT_NOMINAL for standard process variations (classes 3 and 4), "
                        "DEFER_RARE_REVIEW for rare/anomalous defect morphologies (classes 1, 2, 5) requiring human review, "
                        "or NO_DEFECT_REIMAGE for blank or misaligned scans (class 6)."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Decision confidence score between 0.0 and 1.0.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Concise engineering rationale based on morphology, retrieved neighbors, and mask quality.",
                },
                "cited_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific evidence points cited from nearest neighbors, morphology, and baseline scores.",
                },
            },
            "required": ["decision", "confidence", "rationale", "cited_evidence"],
        },
    },
}


def resolve_api_key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("META_API_KEY=") or line.startswith("MODEL_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and not val.startswith("your_"):
                    key = val.strip("'\"")
                    break
    if not key:
        raise ValueError("META_API_KEY (or MODEL_API_KEY) must be set in environment or .env")
    return key


@dataclass
class TriageAgentResult:
    """Complete structured output from one image triage evaluation."""

    image_id: str
    decision: str  # "ACCEPT_NOMINAL" | "DEFER_RARE_REVIEW" | "NO_DEFECT_REIMAGE"
    confidence: float
    rationale: str
    cited_evidence: List[str]
    is_empty_scan: bool
    empty_scan_reason: Optional[str]

    # Segmentation & Verification
    initial_box: Optional[List[int]]
    refined_box: Optional[List[int]]
    initial_unet_sam_iou: Optional[float]
    final_unet_sam_iou: Optional[float]
    refinement_triggered: bool
    refinement_accepted: bool
    refinement_strategy: str
    refinement_reasons: List[str]
    delta_iou_proxy: float

    # Baselines
    baseline1: BaselineVerdict
    baseline2: BaselineVerdict
    baseline3: BaselineVerdict

    # Nearest Neighbors
    retrieved_neighbors: List[Dict[str, Any]]

    # Telemetry
    sam_response_id: Optional[str]
    sam_initial_response_id: Optional[str]
    sam_refined_response_id: Optional[str]
    triage_response_id: Optional[str]
    triage_model: str
    token_usage: Optional[Dict[str, Any]]

    # Full binary masks for outer GT evaluation (excluded from JSON serialization)
    initial_sam_mask: Optional[np.ndarray] = None
    final_sam_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("initial_sam_mask", None)
        d.pop("final_sam_mask", None)
        d["baseline1"] = asdict(self.baseline1)
        d["baseline2"] = asdict(self.baseline2)
        d["baseline3"] = asdict(self.baseline3)
        return d


class VerifiableTriageAgent:
    """Verifiable Multimodal Triage Agent for Industrial SEM defect classification."""

    def __init__(
        self,
        retrieval_bank: RetrievalBank,
        unet_model_path: Path | str = "outputs/gate1/checkpoints/best_unet.pt",
        triage_model: str = "muse-spark-1.3",
        api_base_url: str = "https://api.meta.ai/v1",
        api_key: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.retrieval_bank = retrieval_bank
        self.triage_model = triage_model
        self.api_base_url = api_base_url
        self.api_key = api_key or resolve_api_key()
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base_url, timeout=120.0)

        # Baseline evaluators
        self.b1_evaluator = Baseline1Confidence(confidence_cutoff=0.90)
        self.b2_evaluator = Baseline2Discordance(discordance_cutoff=0.35)
        self.b3_evaluator = Baseline3KNNLogistic(rare_prob_cutoff=0.50)

        # Local U-Net for ground-truth-free mask verification
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.unet: Optional[ResNet18UNet] = None
        unet_p = Path(unet_model_path)
        if unet_p.exists():
            cp = torch.load(unet_p, map_location=self.device, weights_only=True)
            self.unet = ResNet18UNet().to(self.device)
            self.unet.load_state_dict(cp)
            self.unet.eval()

    def _predict_unet_mask(self, img_pil: Image.Image) -> np.ndarray:
        """Locally predicts 480x480 U-Net binary probability mask (prob >= 0.5, border zeroed)."""
        if self.unet is None:
            # Fallback if checkpoint missing: empty canvas
            return np.zeros((480, 480), dtype=np.uint8)

        img_arr = np.array(img_pil.convert("L"), dtype=np.float32) / 255.0
        with torch.no_grad():
            x = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.unet(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = (prob >= 0.5).astype(np.uint8)
        mask[:, 477:] = 0
        return mask

    def _create_visual_artifacts(
        self,
        img_pil: Image.Image,
        sam_mask: np.ndarray,
        box: List[int],
        neighbors: List[RetrievedNeighbor],
        data_dir: Path,
    ) -> Tuple[str, str]:
        """Creates base64 data URLs for: (1) query crop with mask overlay, (2) Top-5 neighbor strip."""
        # 1. Query defect crop with overlay
        x1, y1, x2, y2 = [int(v) for v in box]
        pad_x = max(16, int(0.20 * (x2 - x1 + 1)))
        pad_y = max(16, int(0.20 * (y2 - y1 + 1)))
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(480, x2 + pad_x + 1), min(480, y2 + pad_y + 1)

        crop_rgb = img_pil.convert("RGB").crop((cx1, cy1, cx2, cy2))
        crop_mask = sam_mask[cy1:cy2, cx1:cx2]

        # Draw semi-transparent cyan overlay on mask
        overlay_arr = np.array(crop_rgb)
        cyan_color = np.array([0, 220, 220], dtype=np.uint8)
        mask_idx = crop_mask > 0
        overlay_arr[mask_idx] = (0.5 * overlay_arr[mask_idx] + 0.5 * cyan_color).astype(np.uint8)
        query_crop_img = Image.fromarray(overlay_arr)

        # Convert to data URL
        q_buf = io.BytesIO()
        query_crop_img.save(q_buf, format="PNG")
        query_data_url = f"data:image/png;base64,{base64.b64encode(q_buf.getvalue()).decode('ascii')}"

        # 2. Horizontal strip of Top-5 retrieved neighbors
        tile_w, tile_h = 100, 100
        strip_img = Image.new("RGB", (tile_w * len(neighbors), tile_h + 24), color=(30, 30, 30))
        draw = ImageDraw.Draw(strip_img)

        for i, n in enumerate(neighbors):
            n_img_p = data_dir / "images" / f"{n.id}.jpg"
            if n_img_p.exists():
                n_pil = Image.open(n_img_p).convert("L")
                bx = n.box
                n_crop = n_pil.crop((bx[0], bx[1], bx[2] + 1, bx[3] + 1)).resize((tile_w, tile_h))
                strip_img.paste(n_crop.convert("RGB"), (i * tile_w, 24))
            # Text label: rank, class, dist
            lbl_text = f"#{n.rank} {n.class_name} (d={n.distance:.2f})"
            draw.text((i * tile_w + 4, 4), lbl_text, fill=(240, 240, 240))

        s_buf = io.BytesIO()
        strip_img.save(s_buf, format="PNG")
        strip_data_url = f"data:image/png;base64,{base64.b64encode(s_buf.getvalue()).decode('ascii')}"

        return query_data_url, strip_data_url

    def triage_image(
        self,
        image_path: Path | str,
        unet_pred_entry: Dict[str, Any],
        data_dir: Path | str = "data/carinthia_s",
        box_override: Optional[List[int]] = None,
    ) -> TriageAgentResult:
        """Evaluates one image according to the verifiable triage protocol.

        Args:
            image_path: Path to the 480x480 SEM image.
            unet_pred_entry: U-Net prediction entry from results/gate1/unet_predicted_boxes.json.
            data_dir: Directory containing carinthia_s images/ and masks/.
            box_override: Optional bounding box override (used for perturbation testing in dry-run).

        Returns:
            TriageAgentResult containing all verdicts, proxies, and telemetry.
        """
        img_p = Path(image_path)
        data_p = Path(data_dir)
        sid = unet_pred_entry["id"]
        img_pil = Image.open(img_p).convert("L")

        # -------------------------------------------------------------------
        # Step 1: Condition C3 Deterministic Empty-Scan Gate
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # Step 2: SAM Segmentation & Condition C1 Deterministic Mask Verification
        # -------------------------------------------------------------------
        target_box = box_override or unet_pred_entry.get("largest_component_bbox")
        if target_box is None:
            target_box = [0, 0, 476, 479]
        # Clamp coordinates to valid image bounds (x <= 476)
        bx1 = max(0, min(476, int(target_box[0])))
        by1 = max(0, min(479, int(target_box[1])))
        bx2 = max(0, min(476, int(target_box[2])))
        by2 = max(0, min(479, int(target_box[3])))
        active_box = [bx1, by1, bx2, by2]

        # Generate local U-Net prediction for GT-free proxy verification
        m_unet = self._predict_unet_mask(img_pil)

        # Base64 encode full image for SAM API call
        img_b64 = base64.b64encode(img_p.read_bytes()).decode("ascii")
        full_data_url = f"data:image/jpeg;base64,{img_b64}"

        # Initial prompt to SAM 3.1
        init_prompt = sam_decoder.build_box_prompt(
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
        parsed_sam = sam_decoder.decode_output_text(getattr(sam_resp, "output_text", "") or "")
        if parsed_sam.masks:
            m_sam_init = sam_decoder.mask_to_full_canvas(parsed_sam.masks[0], 480, 480)
        else:
            m_sam_init = np.zeros((480, 480), dtype=np.uint8)

        # Condition C1 Mask Verification
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
            refinement_strategy = "composite_box_and_point"
            # Re-prompt SAM using U-Net core box + positive interior point prompt
            unet_box = unet_pred_entry.get("largest_component_bbox") or active_box
            ux1 = max(0, min(476, int(unet_box[0])))
            uy1 = max(0, min(479, int(unet_box[1])))
            ux2 = max(0, min(476, int(unet_box[2])))
            uy2 = max(0, min(479, int(unet_box[3])))
            core_box = [ux1, uy1, ux2, uy2]

            # Interior positive point (centroid of U-Net core box)
            pt_x = (core_box[0] + core_box[2]) // 2
            pt_y = (core_box[1] + core_box[3]) // 2

            ref_obj = msp.PromptObject(
                id=1,
                prompts=[
                    msp.FramePrompt(
                        frame=0,
                        box=msp.PromptBox(left=core_box[0], top=core_box[1], right=core_box[2] + 1, bottom=core_box[3] + 1),
                        points=[msp.PromptPoint(x=pt_x, y=pt_y, label="positive")],
                    )
                ],
            )
            refined_prompt = msp.build_object_prompt(size=(480, 480), objects=[ref_obj])

            sam_resp_ref = self.client.responses.create(
                model="sam-3.1",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": refined_prompt},
                        {"type": "input_image", "image_url": full_data_url},
                    ],
                }],
            )
            sam_resp_ref_id = getattr(sam_resp_ref, "id", None)
            parsed_ref = sam_decoder.decode_output_text(getattr(sam_resp_ref, "output_text", "") or "")
            if parsed_ref.masks:
                m_sam_ref = sam_decoder.mask_to_full_canvas(parsed_ref.masks[0], 480, 480)
                ref_verif = verify_sam_mask(m_unet, m_sam_ref, core_box)
                refined_iou = ref_verif.unet_sam_iou
                # Adopt refined mask if agreement improved
                if refined_iou > init_iou:
                    m_sam_final = m_sam_ref
                    final_iou = refined_iou
                    final_box = core_box
                    delta_iou_proxy = refined_iou - init_iou
                    refinement_accepted = True
                else:
                    refinement_accepted = False
        else:
            # Locked immediately: monotonicity guardrail guarantees delta_iou = 0.0
            delta_iou_proxy = 0.0

        # -------------------------------------------------------------------
        # Step 3: Morphology Extraction & Unbiased Top-5 Retrieval
        # -------------------------------------------------------------------
        morph = compute_defect_morphology(img_pil, m_sam_final, final_box)
        query_vec = morph.to_vector()

        # Retrieve Top-5 neighbors from nominal_train + rare_dev with Leave-One-Out
        neighbors = self.retrieval_bank.retrieve_top_k(query_vec, k=5, exclude_id=sid)
        std_feat = self.retrieval_bank.standardize(query_vec)

        # -------------------------------------------------------------------
        # Step 4: Evaluate Non-LLM Baselines
        # -------------------------------------------------------------------
        b1 = self.b1_evaluator.evaluate(unet_pred_entry, is_empty_scan=False)
        b2 = self.b2_evaluator.evaluate(final_iou, is_empty_scan=False)
        b3 = self.b3_evaluator.evaluate(neighbors, std_feat, is_empty_scan=False)

        # -------------------------------------------------------------------
        # Step 5: Visual Artifact Synthesis
        # -------------------------------------------------------------------
        q_data_url, strip_data_url = self._create_visual_artifacts(
            img_pil, m_sam_final, final_box, neighbors, data_p
        )

        # -------------------------------------------------------------------
        # Step 6: Multimodal Decision Node (muse-spark-1.3 / muse-spark-open-1026)
        # -------------------------------------------------------------------
        # Format retrieval table
        neighbor_table_lines = [
            "| Rank | Exemplar ID | Category | Distance | Area | Aspect Ratio |",
            "|---|---|---|---|---|---|",
        ]
        for n in neighbors:
            m = n.morphology
            neighbor_table_lines.append(
                f"| #{n.rank} | {n.id[:8]}.. | {n.class_name} | {n.distance:.3f} | {m.get('area', 0)}px | {m.get('aspect_ratio', 0.0):.2f} |"
            )
        neighbor_table_str = "\n".join(neighbor_table_lines)

        prompt_text = (
            f"You are the Lead Semiconductor Defect Triage Agent inspecting an industrial wafer SEM image.\n\n"
            f"### Query Defect Quantitative Metrics:\n"
            f"- Bounding Box: {final_box}\n"
            f"- Segmented Area: {morph.area} pixels\n"
            f"- Aspect Ratio: {morph.aspect_ratio:.2f} (w/h)\n"
            f"- Box Fill Ratio: {morph.box_fill_ratio:.2f}\n"
            f"- U-Net/SAM Agreement (Proxy IoU): {final_iou:.3f}\n"
            f"- Mean Intensity: {morph.crop_mean_intensity:.2f}, Std: {morph.crop_std_intensity:.2f}\n\n"
            f"### Top-5 Retrieved Exemplars (Leave-One-Out Training Bank):\n"
            f"{neighbor_table_str}\n\n"
            f"### Non-LLM Baseline Indicators:\n"
            f"- Baseline 1 (U-Net Max Confidence): {b1.decision} (max_prob={b1.score:.3f})\n"
            f"- Baseline 2 (U-Net/SAM Discordance): {b2.decision} (discordance={b2.score:.3f})\n"
            f"- Baseline 3 (k-NN & Logistic Scorer): {b3.decision} (P(rare)={b3.score:.3f})\n\n"
            f"### Triage Decision Protocol:\n"
            f"- If the defect matches nominal process variations (class_3 or class_4): commit 'ACCEPT_NOMINAL'.\n"
            f"- If the defect matches rare/anomalous defect morphologies (class_1, class_2, or class_5) or displays high discordance with nominal patterns: commit 'DEFER_RARE_REVIEW'.\n"
            f"- If the scan shows no defect / blank wafer: commit 'NO_DEFECT_REIMAGE'.\n\n"
            f"Call commit_triage_decision with your final decision, confidence, engineering rationale, and cited evidence."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": q_data_url}},
                    {"type": "image_url", "image_url": {"url": strip_data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        triage_resp = self.client.chat.completions.create(
            model=self.triage_model,
            messages=messages,
            tools=[COMMIT_TOOL],
            tool_choice="auto",
            temperature=0,
        )

        triage_resp_id = triage_resp.id
        token_usage = json.loads(json.dumps(triage_resp.usage.model_dump())) if triage_resp.usage else None

        # Parse tool call
        decision = "DEFER_RARE_REVIEW"  # Safe default on parse error
        confidence = 0.5
        rationale = "No tool call returned"
        cited_evidence: List[str] = []

        tc = triage_resp.choices[0].message.tool_calls
        if tc:
            try:
                args = json.loads(tc[0].function.arguments)
                decision = args.get("decision", decision)
                confidence = float(args.get("confidence", confidence))
                rationale = args.get("rationale", rationale)
                cited_evidence = args.get("cited_evidence", cited_evidence)
            except Exception as e:
                rationale = f"Tool call argument parse error: {e}"
        else:
            # Fallback parsing from content
            content = triage_resp.choices[0].message.content or ""
            if "ACCEPT_NOMINAL" in content:
                decision = "ACCEPT_NOMINAL"
            elif "NO_DEFECT_REIMAGE" in content:
                decision = "NO_DEFECT_REIMAGE"
            else:
                decision = "DEFER_RARE_REVIEW"
            rationale = content[:200]

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
            initial_unet_sam_iou=init_iou,
            final_unet_sam_iou=final_iou,
            refinement_triggered=verif.triggered_refinement,
            refinement_accepted=refinement_accepted,
            refinement_strategy=refinement_strategy,
            refinement_reasons=verif.trigger_reasons,
            delta_iou_proxy=delta_iou_proxy,
            baseline1=b1,
            baseline2=b2,
            baseline3=b3,
            retrieved_neighbors=[asdict(n) for n in neighbors],
            sam_response_id=sam_resp_id,
            sam_initial_response_id=sam_resp_id,
            sam_refined_response_id=sam_resp_ref_id,
            triage_response_id=triage_resp_id,
            triage_model=self.triage_model,
            token_usage=token_usage,
            initial_sam_mask=m_sam_init,
            final_sam_mask=m_sam_final,
        )
