def test_import_modules():
    """Verifies that all library modules import cleanly."""
    import scripts.train_unet
    import src
    import src.api
    import src.api.sam_decoder
    import src.dataset
    import src.dataset.carinthia_loader
    import src.dataset.carinthia_s
    import src.dataset.loader

    assert hasattr(src.api.sam_decoder, "run_self_test")
    assert hasattr(src.dataset.carinthia_loader.CarinthiaSDataset, "compute_bounding_box")
    assert hasattr(src.dataset.carinthia_loader.CarinthiaSDataset, "preprocess_mask")
    assert hasattr(src.dataset.loader, "CarinthiaLoader")
    assert hasattr(src.dataset.loader, "preprocess_mask")
    assert hasattr(src.dataset.loader, "compute_bounding_box")
    assert hasattr(scripts.train_unet, "ResNet18UNet")
    assert hasattr(scripts.train_unet, "load_unet_checkpoint")
    assert hasattr(scripts.train_unet, "evaluate_split")
