def test_import_modules():
    """Verifies that all library modules import cleanly."""
    import src
    import src.api
    import src.api.sam_decoder
    import src.dataset
    import src.dataset.carinthia_loader

    assert hasattr(src.api.sam_decoder, "run_self_test")
    assert hasattr(src.dataset.carinthia_loader.CarinthiaSDataset, "compute_bounding_box")
    assert hasattr(src.dataset.carinthia_loader.CarinthiaSDataset, "preprocess_mask")
