"""Unit tests for Verifiable Triage Agent components."""

import numpy as np
from PIL import Image

from src.agent.baselines import (
    Baseline1Confidence,
    Baseline2Discordance,
    Baseline3KNNLogistic,
)
from src.agent.morphology import compute_defect_morphology
from src.agent.retrieval import RetrievalBank, RetrievedNeighbor
from src.agent.triage_agent import extract_interior_points
from src.agent.verifier import evaluate_empty_scan_gate, verify_sam_mask


def test_empty_scan_gate():
    # Empty scan (class 6 pattern)
    empty_entry = {
        "id": "c6_test",
        "predicted_foreground_pixels_raw": 10,
        "predicted_foreground_pixels_filtered": 0,
        "largest_component_bbox": None,
    }
    res = evaluate_empty_scan_gate(empty_entry, threshold_pixels=64)
    assert res.is_empty is True
    assert "No connected foreground component" in res.reason

    # Below threshold
    dust_entry = {
        "id": "dust_test",
        "predicted_foreground_pixels_raw": 30,
        "predicted_foreground_pixels_filtered": 22,
        "largest_component_bbox": [472, 471, 476, 477],
    }
    res_dust = evaluate_empty_scan_gate(dust_entry, threshold_pixels=64)
    assert res_dust.is_empty is True

    # Valid defect
    valid_entry = {
        "id": "defect_test",
        "predicted_foreground_pixels_raw": 2400,
        "predicted_foreground_pixels_filtered": 2400,
        "largest_component_bbox": [200, 200, 280, 280],
    }
    res_valid = evaluate_empty_scan_gate(valid_entry, threshold_pixels=64)
    assert res_valid.is_empty is False


def test_mask_verification_and_monotonicity():
    # Identical masks -> high agreement, must be locked immediately
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:60, 20:60] = 1
    box = [20, 20, 59, 59]

    verif = verify_sam_mask(mask1, mask1, box)
    assert verif.unet_sam_iou == 1.0
    assert verif.is_locked is True
    assert verif.triggered_refinement is False

    # Disagreeing masks -> must trigger refinement
    mask_bad = np.zeros((100, 100), dtype=np.uint8)
    mask_bad[80:95, 80:95] = 1

    verif_bad = verify_sam_mask(mask1, mask_bad, box)
    assert verif_bad.unet_sam_iou < 0.65
    assert verif_bad.triggered_refinement is True
    assert verif_bad.is_locked is False


def test_morphology_and_retrieval_loo():
    # Synthetic image & mask
    img = Image.new("L", (480, 480), color=120)
    mask = np.zeros((480, 480), dtype=np.uint8)
    mask[100:150, 100:200] = 1
    box = [100, 100, 199, 149]

    morph = compute_defect_morphology(img, mask, box)
    assert morph.area == 50 * 100
    assert morph.aspect_ratio > 1.5
    vec = morph.to_vector()
    assert len(vec) == 16

    # Test LOO retrieval
    samples = [
        {"id": "sample_1", "class_label": 3, "box": box, "morphology": morph.to_dict()},
        {"id": "sample_2", "class_label": 1, "box": box, "morphology": morph.to_dict()},
        {"id": "sample_3", "class_label": 4, "box": box, "morphology": morph.to_dict()},
    ]
    vecs = np.zeros((3, 16), dtype=np.float32)
    vecs[0, :] = 1.0
    vecs[1, :] = 2.0
    vecs[2, :] = 3.0
    bank = RetrievalBank(
        samples=samples,
        vectors=vecs,
        mean=np.zeros(16, dtype=np.float32),
        std=np.ones(16, dtype=np.float32),
    )

    # Without LOO: closest to 1.0 is sample_1
    top_no_loo = bank.retrieve_top_k(np.ones(16, dtype=np.float32), k=2)
    assert top_no_loo[0].id == "sample_1"

    # With LOO excluding sample_1: closest is sample_2
    top_loo = bank.retrieve_top_k(np.ones(16, dtype=np.float32), k=2, exclude_id="sample_1")
    assert top_loo[0].id == "sample_2"


def test_baselines():
    b1 = Baseline1Confidence(confidence_cutoff=0.90)
    b2 = Baseline2Discordance(discordance_cutoff=0.35)
    b3 = Baseline3KNNLogistic(rare_prob_cutoff=0.50)

    # Empty scan handling
    v1_emp = b1.evaluate({}, is_empty_scan=True)
    assert v1_emp.decision == "NO_DEFECT_REIMAGE"

    v2_emp = b2.evaluate(0.0, is_empty_scan=True)
    assert v2_emp.decision == "NO_DEFECT_REIMAGE"

    v3_emp = b3.evaluate([], None, is_empty_scan=True)
    assert v3_emp.decision == "NO_DEFECT_REIMAGE"

    # Active defect handling
    v1 = b1.evaluate({"max_prob": 0.98}, is_empty_scan=False)
    assert v1.decision == "ACCEPT_NOMINAL"

    v2 = b2.evaluate(0.40, is_empty_scan=False)  # discordance = 0.60 >= 0.35
    assert v2.decision == "DEFER_RARE_REVIEW"

    # k-NN rare decision
    rare_neighbors = [
        RetrievedNeighbor(rank=1, id="n1", class_label=1, class_name="class_1", distance=0.1, box=[0, 0, 10, 10], morphology={}),
        RetrievedNeighbor(rank=2, id="n2", class_label=5, class_name="class_5", distance=0.2, box=[0, 0, 10, 10], morphology={}),
        RetrievedNeighbor(rank=3, id="n3", class_label=3, class_name="class_3", distance=1.5, box=[0, 0, 10, 10], morphology={}),
    ]
    v3 = b3.evaluate(rare_neighbors, None, is_empty_scan=False)
    assert v3.decision == "DEFER_RARE_REVIEW"
    assert v3.score > 0.50


def test_extract_interior_points():
    # Empty mask
    empty = np.zeros((50, 50), dtype=np.uint8)
    assert extract_interior_points(empty) == []

    # Solid 30x30 square centered in 50x50
    solid = np.zeros((50, 50), dtype=np.uint8)
    solid[10:40, 10:40] = 1
    pts = extract_interior_points(solid, max_points=2)
    assert len(pts) > 0
    for px, py in pts:
        # Must be well inside the interior, not on the edges
        assert 15 <= px <= 35
        assert 15 <= py <= 35


def test_leak_detection_triggers_refinement():
    # UNet area = 100 px, SAM area = 200 px (> 1.30 ratio)
    m_unet = np.zeros((100, 100), dtype=np.uint8)
    m_unet[20:30, 20:30] = 1  # 100 px

    m_sam = np.zeros((100, 100), dtype=np.uint8)
    m_sam[20:30, 20:40] = 1  # 200 px
    box = [20, 20, 29, 39]

    verif = verify_sam_mask(m_unet, m_sam, box)
    assert verif.area_ratio == 2.0
    assert verif.triggered_refinement is True
    assert any("leak" in r.lower() or "over-segmentation" in r.lower() for r in verif.trigger_reasons)
