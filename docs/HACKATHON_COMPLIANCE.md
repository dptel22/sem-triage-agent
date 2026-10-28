# Hackathon Rules & Academic Integrity Compliance

## 1. Participation & Scope Declaration
- **Participant:** Solo participant project.
- **Repository Visibility:** Currently private during active development; will be made publicly accessible upon freezing before submission deadline.
- **New-Work Policy:** 100% of the project code, evaluation harnesses, U-Net training pipeline, and triage agent implementations were authored entirely during the designated hackathon event window. Zero pre-existing proprietary codebase was imported or copied.
- **Git History Integrity:** All commits in git history reflect chronological, immutable progress across branches (`feat/gate0-diagnostics`, `feat/gate1-unet-baseline`, `chore/merge-gates`, `docs/professional-repo`). No force-pushes, commit squashes, or backdated commits were executed.

---

## 2. Component Lineage: Authored vs. Third-Party

| Component / Asset | Origin | License / Terms | Role in Project | Authored During Event? |
| :--- | :--- | :--- | :--- | :---: |
| **Carinthia-S Dataset** | Corinna Kofler, Vahidin Hasić, et al. ([DOI: 10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427)) | Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) | Raw SEM defect images and expert binary segmentation masks | **No (Dataset Benchmark)** |
| **Meta Model API** | Meta AI (`https://api.meta.ai/v1`) | Meta API Terms of Service | Hosted foundation models: `sam-3.1` (vision) and `muse-spark-1.3` (multimodal reasoning) | **No (External Cloud API)** |
| **`meta-sam-parser`** | Meta Models ([github.com/meta-models/meta-sam](https://github.com/meta-models/meta-sam)) | MIT License | Official reference decoder for SAM 3 base85/arithmetic wire format | **No (Third-Party Library)** |
| **PyTorch & Torchvision** | Linux Foundation / Meta | BSD-3-Clause | Deep learning framework for local U-Net baseline training | **No (Standard Framework)** |
| **OpenCV (`opencv-python-headless`)** | OpenCV Team | Apache-2.0 | Connected-component analysis and morphological filtering | **No (Standard Library)** |
| **SAM Decoder Wrapper (`src/api/sam_decoder.py`)** | Authored for this project | Apache-2.0 / MIT | Bit-exact prompt construction, half-open coordinate mapping, and mask canvas assembly | **YES (Authored in Gate 0)** |
| **Dataset Loader & Border Cleaner (`src/dataset/carinthia_loader.py`)** | Authored for this project | Apache-2.0 / MIT | Preprocessing, 3px right-border zeroing, and inclusive/half-open bbox generation | **YES (Authored in Gate 1)** |
| **Leakage-Free Dataset Splitter (`scripts/create_gate1_splits.py`)** | Authored for this project | Apache-2.0 / MIT | Deterministic 7-way stratified split generator with LOO indexing and Gate 0 isolation | **YES (Authored in Gate 1)** |
| **U-Net Proposer Pipeline (`scripts/train_unet_baseline.py`)** | Authored for this project | Apache-2.0 / MIT | ResNet18 U-Net training, mixed precision, and connected-component non-oracle bbox export | **YES (Authored in Gate 1)** |
| **Sharpened Test Harness & Manifest (`docs/sharpened_test_protocol.md`)** | Authored for this project | Apache-2.0 / MIT | Pre-registered objective evaluation harness with frozen seed 42 sample IDs | **YES (Authored in Gate 1)** |

---

## 3. Data Rights & Academic Attribution
- **Dataset Title:** Carinthia-S dataset
- **Authors:** Corinna Kofler (KAI GmbH), Vahidin Hasić (University of Sarajevo), et al.
- **Persistent Identifier (DOI):** [10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Compliance Action:** The dataset is properly attributed across documentation and manifests. No raw binary dataset images or proprietary weights are committed to the git repository; all data remains in gitignored `data/` and `outputs/`.
