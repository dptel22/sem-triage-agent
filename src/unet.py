"""ResNet18-UNet Defect Segmentation Model Architecture and Inference.

Provides single-channel grayscale SEM defect proposal, connected component
analysis, and golden-verified bounding box extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    nn = object  # type: ignore

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


if _TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        """Two 3x3 Conv + BatchNorm + ReLU blocks."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class ResNet18UNet(nn.Module):
        """ResNet18 encoder with U-Net skip-connection decoder for 480x480 SEM segmentation."""

        def __init__(self, in_channels: int = 1, num_classes: int = 1):
            super().__init__()
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            # Adapt first conv for 1-channel grayscale SEM
            self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.init_conv.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool

            self.enc1 = resnet.layer1  # 64 ch, /4
            self.enc2 = resnet.layer2  # 128 ch, /8
            self.enc3 = resnet.layer3  # 256 ch, /16
            self.enc4 = resnet.layer4  # 512 ch, /32

            # Decoder with skip connections
            self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec4 = ConvBlock(512, 256)

            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = ConvBlock(256, 128)

            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = ConvBlock(128, 64)

            self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.dec1 = ConvBlock(128, 64)

            self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec0 = ConvBlock(32, 32)

            self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e0 = self.relu(self.bn1(self.init_conv(x)))  # (B, 64, 240, 240)
            p0 = self.maxpool(e0)                         # (B, 64, 120, 120)
            e1 = self.enc1(p0)                            # (B, 64, 120, 120)
            e2 = self.enc2(e1)                            # (B, 128, 60, 60)
            e3 = self.enc3(e2)                            # (B, 256, 30, 30)
            e4 = self.enc4(e3)                            # (B, 512, 15, 15)

            d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))  # (B, 256, 30, 30)
            d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))  # (B, 128, 60, 60)
            d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))  # (B, 64, 120, 120)
            d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))  # (B, 64, 240, 240)
            d0 = self.dec0(self.up0(d1))                           # (B, 32, 480, 480)
            logits = self.head(d0)                                # (B, 1, 480, 480)
            return logits


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Combined Binary Cross-Entropy and Soft Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()
    return bce + dice_loss


def load_unet(
    checkpoint_path: str | Path,
    device: Optional[torch.device | str] = None,
) -> ResNet18UNet:
    """Loads frozen U-Net model from checkpoint."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load U-Net")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ResNet18UNet().to(dev)
    cp = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(cp)
    model.eval()
    return model


def predict_prob(
    model: ResNet18UNet,
    image: Image.Image | np.ndarray,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """Infers continuous sigmoid probability map with 3px right border zeroed."""
    dev = torch.device(device or next(model.parameters()).device)
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    else:
        img_arr = np.array(image, dtype=np.float32)
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

    with torch.no_grad():
        x = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(dev)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Zero out 3px right border artifact
    prob[:, 477:] = 0.0
    return prob


def predict_mask(
    model: ResNet18UNet,
    image: Image.Image | np.ndarray,
    device: Optional[torch.device | str] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Infers binary segmentation mask (prob >= threshold)."""
    prob = predict_prob(model, image, device)
    return (prob >= threshold).astype(np.uint8)


def extract_component_bboxes(
    binary_mask: np.ndarray,
    min_area: int = 16,
) -> Dict[str, Any]:
    """Extracts connected component bounding boxes, largest component, and filtered foreground bbox.

    Addresses the 'corner bug' / single-pixel border artifact from Gate 0:
    - Eliminates isolated pixels (< min_area) that stretched the oracle bounding boxes.
    - Computes largest_component_bbox and filtered_foreground_bbox.
    - All bboxes follow inclusive [x1, y1, x2, y2] convention.
    """
    total_fg_pixels = int(binary_mask.sum())
    if total_fg_pixels == 0:
        return {
            "overall_bbox": None,
            "largest_component_bbox": None,
            "filtered_foreground_bbox": None,
            "component_boxes": [],
            "filtered_pixel_count": 0,
        }

    # Raw overall bbox (inclusive)
    y_idx, x_idx = np.where(binary_mask > 0)
    overall_bbox = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())]

    if not _CV2_AVAILABLE:
        # Fallback if cv2 missing
        return {
            "overall_bbox": overall_bbox,
            "largest_component_bbox": overall_bbox,
            "filtered_foreground_bbox": overall_bbox,
            "component_boxes": [{"bbox": overall_bbox, "area": total_fg_pixels}],
            "filtered_pixel_count": total_fg_pixels,
        }

    # Connected component analysis (8-connectivity)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    component_boxes = []
    largest_box = None
    largest_area = 0

    valid_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for lbl_idx in range(1, num_labels):
        area = int(stats[lbl_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            x1 = int(stats[lbl_idx, cv2.CC_STAT_LEFT])
            y1 = int(stats[lbl_idx, cv2.CC_STAT_TOP])
            x2 = int(x1 + stats[lbl_idx, cv2.CC_STAT_WIDTH] - 1)
            y2 = int(y1 + stats[lbl_idx, cv2.CC_STAT_HEIGHT] - 1)
            bbox = [x1, y1, x2, y2]
            component_boxes.append({"bbox": bbox, "area": area})
            valid_mask[labels == lbl_idx] = 1

            if area > largest_area:
                largest_area = area
                largest_box = bbox

    # Sort components by area descending
    component_boxes = sorted(component_boxes, key=lambda c: c["area"], reverse=True)

    # Filtered foreground bbox enclosing all valid components
    filtered_pixel_count = int(valid_mask.sum())
    filtered_bbox = None
    if filtered_pixel_count > 0:
        vy_idx, vx_idx = np.where(valid_mask > 0)
        filtered_bbox = [int(vx_idx.min()), int(vy_idx.min()), int(vx_idx.max()), int(vy_idx.max())]
    elif largest_box is not None:
        filtered_bbox = largest_box

    return {
        "overall_bbox": overall_bbox,
        "largest_component_bbox": largest_box,
        "filtered_foreground_bbox": filtered_bbox,
        "component_boxes": component_boxes,
        "filtered_pixel_count": filtered_pixel_count,
    }
