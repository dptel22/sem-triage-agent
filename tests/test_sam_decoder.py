from src.api.sam_decoder import run_self_test, build_box_prompt


def test_sam_decoder_vectors():
    """Runs the official bit-exact verification vectors."""
    assert run_self_test(verbose=False) is True


def test_build_box_prompt():
    """Tests box prompt construction under inclusive and half-open conventions."""
    prompt_half_open = build_box_prompt(10, 20, 30, 40, 480, 480, convention="half_open")
    assert prompt_half_open == "<0f>1<|box;x1=10;y1=20;x2=29;y2=39;w=480;h=480|>"

    prompt_inclusive = build_box_prompt(10, 20, 29, 39, 480, 480, convention="inclusive")
    assert prompt_inclusive == "<0f>1<|box;x1=10;y1=20;x2=29;y2=39;w=480;h=480|>"
