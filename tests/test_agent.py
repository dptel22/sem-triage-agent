"""Unit tests for Verifiable Triage Agent with mocked API (no key required)."""

from unittest.mock import MagicMock
import numpy as np
from PIL import Image

from src.agent.triage_agent import (
    Baseline1Confidence,
    Baseline2Discordance,
    Baseline3KNNLogistic,
    RetrievalBank,
    RetrievedNeighbor,
    VerifiableTriageAgent,
    compute_defect_morphology,
    extract_interior_points,
    resolve_api_key,
)


def test_morphology_and_retrieval_loo():
    """Verifies morphology feature extraction and Leave-One-Out (LOO) candidate exclusion."""
    img = Image.new("L", (480, 480), color=120)
    mask = np.zeros((480, 480), dtype=np.uint8)
    mask[100:150, 100:200] = 1
    box = [100, 100, 199, 149]

    morph = compute_defect_morphology(img, mask, box)
    assert morph.area == 50 * 100
    assert morph.aspect_ratio > 1.5
    vec = morph.to_vector()
    assert len(vec) == 16

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


def test_baselines_logic():
    """Verifies the three deterministic baselines."""
    b1 = Baseline1Confidence(confidence_cutoff=0.90)
    b2 = Baseline2Discordance(discordance_cutoff=0.35)
    b3 = Baseline3KNNLogistic(rare_prob_cutoff=0.50)

    # Empty scan
    assert b1.evaluate({}, is_empty_scan=True).decision == "NO_DEFECT_REIMAGE"
    assert b2.evaluate(0.0, is_empty_scan=True).decision == "NO_DEFECT_REIMAGE"
    assert b3.evaluate([], None, is_empty_scan=True).decision == "NO_DEFECT_REIMAGE"

    # Active defect
    assert b1.evaluate({"max_prob": 0.95}, is_empty_scan=False).decision == "ACCEPT_NOMINAL"
    assert b1.evaluate({"max_prob": 0.75}, is_empty_scan=False).decision == "DEFER_RARE_REVIEW"
    assert b2.evaluate(0.40, is_empty_scan=False).decision == "DEFER_RARE_REVIEW"  # disc = 0.60 >= 0.35

    rare_neighbors = [
        RetrievedNeighbor(rank=1, id="n1", class_label=1, class_name="class_1", distance=0.1, box=[0, 0, 10, 10], morphology={}),
        RetrievedNeighbor(rank=2, id="n2", class_label=2, class_name="class_2", distance=0.2, box=[0, 0, 10, 10], morphology={}),
    ]
    v3 = b3.evaluate(rare_neighbors, None, is_empty_scan=False)
    assert v3.decision == "DEFER_RARE_REVIEW"


def test_extract_interior_points():
    """Verifies iterative erosion extraction of interior negative points."""
    empty = np.zeros((50, 50), dtype=np.uint8)
    assert extract_interior_points(empty) == []

    solid = np.zeros((50, 50), dtype=np.uint8)
    solid[10:40, 10:40] = 1
    pts = extract_interior_points(solid, max_points=2)
    assert len(pts) > 0
    for px, py in pts:
        assert 15 <= px <= 35
        assert 15 <= py <= 35


def test_agent_triage_with_mocked_api(tmp_path):
    """Verifies agent triage loop with mocked SAM and LLM APIs (zero network/token dependency)."""
    # Create synthetic test image
    img_p = tmp_path / "test_sem.jpg"
    Image.new("L", (480, 480), color=128).save(img_p)

    # Mock client
    mock_client = MagicMock()

    # Mock SAM response
    mock_sam_resp = MagicMock()
    mock_sam_resp.id = "mock_sam_resp_123"
    mock_sam_resp.output_text = "<0f>1<|box;x1=100;y1=100;x2=140;y2=140;w=480;h=480|><|mask;x=100;y=100;data=5,5,!!!!!(QO(0lu8?|>"
    mock_client.responses.create.return_value = mock_sam_resp

    # Mock LLM chat completion
    mock_llm_resp = MagicMock()
    mock_llm_resp.id = "mock_chat_456"
    mock_choice = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.function.arguments = '{"decision": "ACCEPT_NOMINAL", "confidence": 0.95, "rationale": "High observable agreement", "cited_evidence": ["Agreement"]}'
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_llm_resp.choices = [mock_choice]
    mock_llm_resp.usage.prompt_tokens = 100
    mock_llm_resp.usage.completion_tokens = 50
    mock_llm_resp.usage.total_tokens = 150
    mock_client.chat.completions.create.return_value = mock_llm_resp

    bank = RetrievalBank(
        samples=[],
        vectors=np.zeros((0, 16), dtype=np.float32),
        mean=np.zeros(16, dtype=np.float32),
        std=np.ones(16, dtype=np.float32),
    )

    agent = VerifiableTriageAgent(
        retrieval_bank=bank,
        unet_model_path=tmp_path / "non_existent.pt",  # runs with fallback empty unet
        client=mock_client,
    )

    unet_entry = {
        "id": "synthetic_01",
        "largest_component_bbox": [100, 100, 140, 140],
        "predicted_foreground_pixels_filtered": 200,
        "predicted_foreground_pixels_raw": 200,
        "max_prob": 0.92,
    }

    res = agent.triage_image(
        image_path=img_p,
        unet_pred_entry=unet_entry,
        data_dir=tmp_path,
    )

    assert res.image_id == "synthetic_01"
    assert res.decision == "ACCEPT_NOMINAL"
    assert res.confidence == 0.95
    assert res.is_empty_scan is False
    assert res.sam_response_id == "mock_sam_resp_123"
    assert res.triage_response_id == "mock_chat_456"

def test_open_weights_provider_routing(monkeypatch):
    """Verifies API key resolution routes correctly for open-weights providers (Baseten/Modal)."""
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_WEIGHTS_API_KEY", raising=False)

    monkeypatch.setenv("BASETEN_API_KEY", "test_baseten_key")
    assert resolve_api_key(provider="baseten") == "test_baseten_key"

    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setenv("META_API_KEY", "test_meta_key")
    assert resolve_api_key(provider="meta") == "test_meta_key"
