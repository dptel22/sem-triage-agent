"""Unit tests for SAM 3.1 wire-format decoder and prompt builders."""


from src.sam import (
    build_box_prompt,
    build_point_prompt,
    decode_output_text,
    mask_to_full_canvas,
    run_self_test,
)


def test_sam_decoder_self_test_vectors():
    """Verifies official Meta verification vectors decode bit-exactly."""
    assert run_self_test(verbose=False) is True


def test_build_box_prompt():
    """Verifies SAM box prompt construction with coordinate conventions."""
    prompt_inc = build_box_prompt(10, 20, 30, 40, 480, 480, convention="inclusive")
    assert prompt_inc == "<0f>1<|box;x1=10;y1=20;x2=30;y2=40;w=480;h=480|>"

    prompt_half = build_box_prompt(10, 20, 30, 40, 480, 480, convention="half_open")
    assert prompt_half == "<0f>1<|box;x1=10;y1=20;x2=29;y2=39;w=480;h=480|>"


def test_build_point_prompt():
    """Verifies SAM point prompt construction."""
    pos_prompt = build_point_prompt(240, 240, label="positive", canvas_w=480, canvas_h=480)
    assert pos_prompt == "<0f>1<|point;x=240;y=240;w=480;h=480|>"


def test_mask_to_full_canvas():
    """Verifies decoded mask record placement on full canvas."""
    line = "<0f>1<|box;x1=10;y1=10;x2=14;y2=14;w=20;h=20|><|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>"
    parsed = decode_output_text(line)
    assert len(parsed.masks) == 1
    canvas = mask_to_full_canvas(parsed.masks[0], canvas_height=20, canvas_width=20)
    assert canvas.shape == (20, 20)
    assert canvas[10:15, 10:15].sum() == parsed.masks[0].raster.sum()
    assert canvas[:10, :].sum() == 0
    assert canvas[15:, :].sum() == 0
