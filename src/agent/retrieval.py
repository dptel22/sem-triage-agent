"""Unbiased Top-5 Retrieval Bank with Leave-One-Out (LOO) support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from src.agent.morphology import compute_defect_morphology
from src.dataset.carinthia_loader import CarinthiaSDataset


@dataclass(frozen=True)
class RetrievedNeighbor:
    """One nearest neighbor exemplar retrieved from the training bank."""

    rank: int
    id: str
    class_label: int  # 1..5
    class_name: str  # "class_1" .. "class_5"
    distance: float
    box: List[int]
    morphology: Dict[str, Any]


class RetrievalBank:
    """Unified retrieval bank indexed over nominal_train (3437) + rare_dev (17)."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        vectors: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.samples = samples
        self.vectors = vectors  # shape (N, D) standardized
        self.mean = mean
        self.std = std
        self.id_to_index = {s["id"]: i for i, s in enumerate(samples)}

    def standardize(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.mean) / np.maximum(self.std, 1e-6)

    def retrieve_top_k(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[RetrievedNeighbor]:
        """Retrieves Top-K nearest neighbors overall using Euclidean distance.

        Supports Leave-One-Out: if exclude_id matches an entry in the bank,
        it is excluded from the candidate search.
        """
        q_std = self.standardize(query_vector)
        # Compute Euclidean distances to all candidates
        dists = np.linalg.norm(self.vectors - q_std, axis=1)

        # Apply Leave-One-Out masking if query ID is present in the bank
        if exclude_id and exclude_id in self.id_to_index:
            ex_idx = self.id_to_index[exclude_id]
            dists[ex_idx] = float("inf")

        # Top-k indices
        top_indices = np.argsort(dists)[:k]

        results: List[RetrievedNeighbor] = []
        for rank, idx in enumerate(top_indices, start=1):
            s = self.samples[idx]
            lbl = s["class_label"]
            results.append(
                RetrievedNeighbor(
                    rank=rank,
                    id=s["id"],
                    class_label=lbl,
                    class_name=f"class_{lbl}",
                    distance=float(dists[idx]),
                    box=s["box"],
                    morphology=s["morphology"],
                )
            )
        return results

    @classmethod
    def load_or_build(
        cls,
        data_dir: Path | str = "data/carinthia_s",
        manifest_path: Path | str = "results/gate1/splits_manifest.json",
        cache_path: Path | str = "outputs/gate2/retrieval_bank.json",
    ) -> "RetrievalBank":
        """Loads precomputed retrieval bank from cache or builds it from disk."""
        cache_p = Path(cache_path)
        if cache_p.exists():
            data = json.loads(cache_p.read_text(encoding="utf-8"))
            samples = data["samples"]
            vectors = np.array(data["vectors"], dtype=np.float32)
            mean = np.array(data["mean"], dtype=np.float32)
            std = np.array(data["std"], dtype=np.float32)
            return cls(samples=samples, vectors=vectors, mean=mean, std=std)

        # Build bank from nominal_train + rare_dev
        data_p = Path(data_dir)
        man_p = Path(manifest_path)
        splits = json.load(open(man_p, encoding="utf-8"))["splits"]

        train_pool = splits["nominal_train"] + splits["rare_dev"]
        samples_out: List[Dict[str, Any]] = []
        raw_vectors: List[np.ndarray] = []

        for item in train_pool:
            sid = item["id"]
            lbl = item["class_label"]
            img_p = data_p / "images" / f"{sid}.jpg"
            mask_p = data_p / "masks" / f"{sid}.png"

            if not img_p.exists() or not mask_p.exists():
                continue

            img = Image.open(img_p).convert("L")
            mask_raw = np.array(Image.open(mask_p).convert("L"))
            bin_mask = CarinthiaSDataset.preprocess_mask(mask_raw, threshold=128, zero_border=True)

            tight_box, padded_box = CarinthiaSDataset.compute_bounding_box(
                mask_raw, pad_ratio=0.0, convention="inclusive", threshold=128, zero_border=True
            )
            if tight_box is None:
                continue

            morph = compute_defect_morphology(img, bin_mask, tight_box)
            vec = morph.to_vector()

            samples_out.append({
                "id": sid,
                "class_label": lbl,
                "box": list(tight_box),
                "morphology": morph.to_dict(),
            })
            raw_vectors.append(vec)

        raw_mat = np.array(raw_vectors, dtype=np.float32)
        mean = np.mean(raw_mat, axis=0)
        std = np.std(raw_mat, axis=0)
        std[std == 0] = 1.0
        std_mat = (raw_mat - mean) / std

        # Save cache
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "samples": samples_out,
            "vectors": std_mat.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        cache_p.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")

        return cls(samples=samples_out, vectors=std_mat, mean=mean, std=std)
