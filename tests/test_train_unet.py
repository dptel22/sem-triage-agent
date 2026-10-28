"""Tests for scripts/train_unet.py entrypoint and inference utilities."""

from pathlib import Path
import pytest
import torch

from scripts.train_unet import (
    ConvBlock,
    ResNet18UNet,
    SEMDataset,
    bce_dice_loss,
    build_boxes_export,
    evaluate_split,
    extract_component_bboxes,
    load_unet_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "gate1" / "checkpoints" / "best_unet.pt"
DATA_DIR = PROJECT_ROOT / "data" / "carinthia_s"


def test_re_exports():
    """Verifies that train_unet re-exports required baseline components."""
    assert ResNet18UNet is not None
    assert ConvBlock is not None
    assert SEMDataset is not None
    assert bce_dice_loss is not None
    assert extract_component_bboxes is not None


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="best_unet.pt checkpoint not found")
def test_load_unet_checkpoint():
    """Verifies loading the frozen best_unet.pt checkpoint."""
    model = load_unet_checkpoint(CHECKPOINT_PATH, device=torch.device("cpu"))
    assert isinstance(model, ResNet18UNet)
    assert not model.training  # Should be in eval mode

    # Check forward pass with dummy 1-channel tensor
    x = torch.zeros((1, 1, 480, 480), dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 480, 480)


@pytest.mark.skipif(not CHECKPOINT_PATH.exists() or not (DATA_DIR / "carinthia-s.csv").exists(), reason="Data or checkpoint missing")
def test_evaluate_split_minimal():
    """Verifies evaluate_split produces valid metrics dict and per-image records."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_unet_checkpoint(CHECKPOINT_PATH, device=device)

    # Evaluate single test sample from dataset
    samples = [{"id": "0017618f77bc4dcfb36823422bd3d04c", "class_label": 3}]
    summary, records = evaluate_split(
        model=model,
        samples=samples,
        data_root=DATA_DIR,
        split_name="test_minimal",
        batch_size=1,
        device=device,
    )

    assert summary["split"] == "test_minimal"
    assert summary["total_images"] == 1
    assert "mean_dice_primary" in summary
    assert "median_iou_primary" in summary
    assert len(records) == 1
    assert records[0]["id"] == "0017618f77bc4dcfb36823422bd3d04c"


def test_build_boxes_export_schema():
    """Verifies boxes export structure adheres to downstream SAM schema."""
    fake_records = [
        {
            "id": "sample_1",
            "class_label": 1,
            "split": "nominal_val",
            "overall_bbox": [10, 10, 50, 50],
            "largest_component_bbox": [10, 10, 50, 50],
            "filtered_foreground_bbox": [10, 10, 50, 50],
            "component_boxes": [{"bbox": [10, 10, 50, 50], "area": 1600}],
            "predicted_pixels": 1600,
            "filtered_pixels": 1600,
            "max_prob": 0.99,
            "iou_primary": 0.95,
            "dice_primary": 0.97,
        }
    ]
    boxes_export = build_boxes_export(fake_records, gate0_ids=["sample_1"])
    assert "metadata" in boxes_export
    assert "gate0_20_predicted_boxes" in boxes_export
    assert len(boxes_export["gate0_20_predicted_boxes"]) == 1
    assert boxes_export["gate0_20_predicted_boxes"][0]["id"] == "sample_1"
