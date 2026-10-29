"""Unit tests for deterministic verifiers (Condition C3 and Condition C1)."""

import numpy as np

from src.agent.verifier import evaluate_empty_scan_gate, verify_sam_mask


def test_empty_scan_gate():
    """Verifies Condition C3: Empty-Scan Gate rules and pixel thresholds."""
    # Class 6 empty scan pattern: no component >= 16px (0 filtered pixels)
    empty_entry = {
        "id": "c6_test",
        "predicted_foreground_pixels_raw": 10,
        "predicted_foreground_pixels_filtered": 0,
        "largest_component_bbox": None,
    }
    res = evaluate_empty_scan_gate(empty_entry, threshold_pixels=16)
    assert res.is_empty is True
    assert "Filtered foreground pixels (0) < 16 px" in res.reason

    # Below 16px threshold
    dust_entry = {
        "id": "dust_test",
        "predicted_foreground_pixels_raw": 30,
        "predicted_foreground_pixels_filtered": 8,
        "largest_component_bbox": [472, 471, 476, 477],
    }
    res_dust = evaluate_empty_scan_gate(dust_entry, threshold_pixels=16)
    assert res_dust.is_empty is True

    # Valid foreground defect
    valid_entry = {
        "id": "defect_test",
        "predicted_foreground_pixels_raw": 2400,
        "predicted_foreground_pixels_filtered": 2400,
        "largest_component_bbox": [200, 200, 280, 280],
    }
    res_valid = evaluate_empty_scan_gate(valid_entry, threshold_pixels=16)
    assert res_valid.is_empty is False


def test_mask_verification_and_monotonicity():
    """Verifies Condition C1: Observable mask agreement and monotonicity lock."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:60, 20:60] = 1
    box = [20, 20, 59, 59]

    # Identical masks -> must lock immediately
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


def test_leak_detection_triggers_refinement():
    """Verifies over-segmentation leak triggers refinement."""
    m_unet = np.zeros((100, 100), dtype=np.uint8)
    m_unet[20:30, 20:30] = 1  # 100 px

    m_sam = np.zeros((100, 100), dtype=np.uint8)
    m_sam[20:30, 20:40] = 1  # 200 px (area ratio = 2.0 > 1.30)
    box = [20, 20, 29, 39]

    verif = verify_sam_mask(m_unet, m_sam, box)
    assert verif.area_ratio == 2.0
    assert verif.triggered_refinement is True


def test_hard_monotonicity_lock():
    """Verifies that unet_sam_iou >= 0.68 with 0.50 <= area_ratio <= 2.0 locks mask unconditionally."""
    m_unet = np.zeros((100, 100), dtype=np.uint8)
    m_sam = np.zeros((100, 100), dtype=np.uint8)
    # 70 pixels overlap, 30 unet-only -> inter=70, union=100 -> IoU = 0.70
    m_unet[10:20, 10:20] = 1  # 100 px
    m_sam[10:20, 10:17] = 1   # 70 px
    # Box area = 50 * 50 = 2500 -> box_fill = 70 / 2500 = 0.028 (< 0.10, which would normally trigger fill ratio alert)
    box = [0, 0, 49, 49]

    verif = verify_sam_mask(m_unet, m_sam, box)
    assert verif.unet_sam_iou == 0.70
    assert 0.50 <= verif.area_ratio <= 2.0
    # Must be locked unconditionally due to Hard Monotonicity Lock
    assert verif.is_locked is True
    assert verif.triggered_refinement is False
    assert len(verif.trigger_reasons) == 0
