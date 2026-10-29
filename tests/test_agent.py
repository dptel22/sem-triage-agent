"""Unit tests for Verifiable Triage Agent with mocked API (no key required)."""

import base64
import io
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
    build_3panel_visual_strip,
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
    assert len(vec) == 18

    samples = [
        {"id": "sample_1", "class_label": 3, "box": box, "morphology": morph.to_dict()},
        {"id": "sample_2", "class_label": 1, "box": box, "morphology": morph.to_dict()},
        {"id": "sample_3", "class_label": 4, "box": box, "morphology": morph.to_dict()},
    ]
    vecs = np.zeros((3, 18), dtype=np.float32)
    vecs[0, :] = 1.0
    vecs[1, :] = 2.0
    vecs[2, :] = 3.0
    bank = RetrievalBank(
        samples=samples,
        vectors=vecs,
        mean=np.zeros(18, dtype=np.float32),
        std=np.ones(18, dtype=np.float32),
    )

    # Without LOO: closest to 1.0 is sample_1
    top_no_loo = bank.retrieve_top_k(np.ones(18, dtype=np.float32), k=2)
    assert top_no_loo[0].id == "sample_1"

    # With LOO excluding sample_1: closest is sample_2
    top_loo = bank.retrieve_top_k(np.ones(18, dtype=np.float32), k=2, exclude_id="sample_1")
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
        vectors=np.zeros((0, 18), dtype=np.float32),
        mean=np.zeros(18, dtype=np.float32),
        std=np.ones(18, dtype=np.float32),
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


def test_build_3panel_visual_strip():
    """Verifies in-memory 3-panel visual inspection strip (480 x 1440 RGB PNG)."""
    raw_img = Image.new("L", (480, 480), color=100)
    final_box = [100, 150, 200, 250]
    m_unet = np.zeros((480, 480), dtype=np.uint8)
    m_unet[120:180, 120:180] = 1
    m_sam = np.zeros((480, 480), dtype=np.uint8)
    m_sam[140:200, 140:200] = 1
    nom_img = Image.new("L", (480, 480), color=180)

    data_url = build_3panel_visual_strip(raw_img, final_box, m_unet, m_sam, nom_img)
    assert data_url.startswith("data:image/png;base64,")

    # Decode and verify dimensions
    b64_data = data_url.split(",", 1)[1]
    decoded_bytes = base64.b64decode(b64_data)
    strip_img = Image.open(io.BytesIO(decoded_bytes))
    assert strip_img.size == (1440, 480)
    assert strip_img.mode == "RGB"

    # Panel 1 (left) must have yellow bounding box pixel
    strip_arr = np.array(strip_img)
    # top-left of box is (100, 150)
    yellow = [255, 255, 0]
    assert np.allclose(strip_arr[150, 100], yellow, atol=10)

    # Panel 2 (center, x offset 480):
    # intersection at [140:180, 140:180] -> x in [620, 660] -> green
    green = [0, 255, 0]
    assert np.allclose(strip_arr[150, 480 + 150], green, atol=10)
    # SAM only at [190, 190] -> red
    red = [255, 0, 0]
    assert np.allclose(strip_arr[190, 480 + 190], red, atol=10)
    # UNet only at [130, 130] -> blue
    blue = [0, 100, 255]
    assert np.allclose(strip_arr[130, 480 + 130], blue, atol=10)


def test_triage_removes_baseline3_from_prompt(tmp_path):
    """Verifies that Baseline 3 predictions are excluded from LLM prompt and 3-panel strip is passed."""
    img_p = tmp_path / "test_sem.jpg"
    Image.new("L", (480, 480), color=128).save(img_p)

    mock_client = MagicMock()
    mock_sam_resp = MagicMock()
    mock_sam_resp.id = "mock_sam_1"
    mock_sam_resp.output_text = "<0f>1<|box;x1=100;y1=100;x2=140;y2=140;w=480;h=480|><|mask;x=100;y=100;data=5,5,!!!!!(QO(0lu8?|>"
    mock_client.responses.create.return_value = mock_sam_resp

    captured_messages = []

    def mock_chat_create(*args, **kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        resp = MagicMock()
        resp.id = "mock_resp_1"
        choice = MagicMock()
        tc = MagicMock()
        tc.function.arguments = '{"decision": "DEFER_RARE_REVIEW", "confidence": 0.90, "rationale": "Signal 3", "cited_evidence": ["Signal 3"]}'
        choice.message.tool_calls = [tc]
        resp.choices = [choice]
        return resp

    mock_client.chat.completions.create.side_effect = mock_chat_create

    bank = RetrievalBank(
        samples=[],
        vectors=np.zeros((0, 18), dtype=np.float32),
        mean=np.zeros(18, dtype=np.float32),
        std=np.ones(18, dtype=np.float32),
    )
    agent = VerifiableTriageAgent(
        retrieval_bank=bank,
        unet_model_path=tmp_path / "non_existent.pt",
        client=mock_client,
    )

    unet_entry = {
        "id": "test_b3_exclusion",
        "largest_component_bbox": [100, 100, 140, 140],
        "predicted_foreground_pixels_filtered": 200,
        "predicted_foreground_pixels_raw": 200,
        "max_prob": 0.85,
    }

    agent.triage_image(image_path=img_p, unet_pred_entry=unet_entry, data_dir=tmp_path)

    # Check user prompt
    user_msg = next(m for m in captured_messages if m["role"] == "user")
    content = user_msg["content"]
    text_content = next(c["text"] for c in content if c["type"] == "text")
    image_content = next(c["image_url"]["url"] for c in content if c["type"] == "image_url")

    # Baseline 3 decision or probability MUST NOT be present
    assert "Baseline 3 Non-LLM Assessment" not in text_content
    assert "P(rare)" not in text_content
    assert "Baseline Recommendations" not in text_content
    assert "k-NN rare fraction" not in text_content

    # Image must be the 3-panel strip (PNG data URL)
    assert image_content.startswith("data:image/png;base64,")


def test_fastapi_endpoints():
    """Smoke tests for FastAPI Metrology Command Center endpoints."""
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)

    # 1. Root endpoint serves index.html
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers["content-type"]
    assert "SEM Metrology & Triage Command Center" in root_resp.text

    # 2. Preset queue returns exactly 14 items with complete metadata
    queue_resp = client.get("/api/preset-queue")
    assert queue_resp.status_code == 200
    queue = queue_resp.json()
    assert len(queue) == 14
    sample_ids = [item["sample_id"] for item in queue]
    assert "52b03c7a_perturbed" in sample_ids
    assert "1bf20f9c_perturbed" in sample_ids
    assert "0c15addf" in sample_ids
    assert "2326638f" in sample_ids

    first = queue[0]
    for key in ("sample_id", "defect_type", "split", "resolution", "is_perturbed", "verdict"):
        assert key in first

    # 3. Image and mask endpoints stream PNG bytes
    img_resp = client.get("/api/image/d1147ea9")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    assert len(img_resp.content) > 0

    mask_resp = client.get("/api/mask/d1147ea9")
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "image/png"
    assert len(mask_resp.content) > 0

    # 4. Inspect endpoint single model & arena_both with 4-gate verification traces
    meta_inspect = client.post("/api/inspect", json={"sample_id": "d1147ea9", "model": "muse-spark-1.3"})
    assert meta_inspect.status_code == 200
    meta_data = meta_inspect.json()
    assert "gates" in meta_data
    assert meta_data["gates"]["gate1"]["status"] == "PASS"
    assert meta_data["gates"]["gate4"]["status"] in ("PASS", "FLAG", "REVISE")
    assert meta_data["triage_response_id"].startswith("chatcmpl-")
    assert len(meta_data["verbatim_rationale"]) > 0

    arena_inspect = client.post("/api/inspect", json={"sample_id": "52b03c7a_perturbed", "model": "arena_both"})
    assert arena_inspect.status_code == 200
    arena_data = arena_inspect.json()
    assert arena_data["model"] == "arena_both"
    assert "meta" in arena_data
    assert "baseten" in arena_data
    assert arena_data["baseten"]["triage_response_id"].startswith("chatcmpl-baseten-")
    assert arena_data["agreement"] is True

    # 5. Benchmark summary comparison metrics
    summary_resp = client.get("/api/benchmark-summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert "sharpened_10_comparison" in summary
    comp = summary["sharpened_10_comparison"]
    assert comp["meta"]["mIoU"] > 0.80
    assert comp["baseten"]["mIoU"] > 0.80
    assert comp["agreement_rate"] == 1.0
    assert "baselines" in comp

