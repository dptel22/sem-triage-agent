# sem-triage-agent

Automated triage and segmentation of Scanning Electron Microscope (SEM) semiconductor wafer-defect images using `sam-3.1` and `muse-spark-1.3` via the Meta Model API.

> **Work In Progress**: Gate 0 (environment verification, dataset ground-truth auditing, and baseline API validation) is complete. The multi-turn triage agent architecture is under active development for Gate 1.

---

## Hackathon Build vs Third-Party Assets

- **Built During Hackathon**:
  - `src/api/sam_decoder.py`: Wire format decoding for SAM-3.1 masks (supporting Base85, one-bit run-length, and lossless modes).
  - `src/dataset/carinthia_loader.py`: Dataset loader and batch sampler for Carinthia-S SEM images and masks.
  - `scripts/run_gate0_api_tests.py`: Gate 0 live API test harness and 4-arm benchmark runner.
  - `scripts/inspect_carinthia_local.py`: Ground-truth dataset auditing and benchmark manifest generator.
  - `scripts/modal_dataset_setup.py`: Modal cloud volume setup and dataset staging automation.
  - `results/gate0/`: Raw API empirical verification and benchmark results.

- **Third-Party Assets & Attribution**:
  - **Carinthia-S Dataset**: Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). DOI: [10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427). Creators: Corinna Kofler (KAI GmbH) and Vahidin Hasić (University of Sarajevo).
  - **Meta Model API**: Hosted API providing access to `sam-3.1` and `muse-spark-1.3`.
  - **meta-sam-parser**: Official reference parser for Meta SAM mask response streams.

---

## Setup & Getting Started

### 1. Installation
Clone the repository and install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and configure your API credentials:
```bash
cp .env.example .env
```
Populate `META_API_KEY` (or `MODEL_API_KEY`) with your Meta Model API key.

### 3. Dataset Setup
The dataset (`data/`) is not tracked in the git repository. You can obtain it either:
- Directly from Zenodo at DOI [10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427) and unpack into `data/carinthia_s/`.
- Or by running the automated Modal volume staging script:
  ```bash
  modal run scripts/modal_dataset_setup.py
  ```

### 4. Running the Self-Test
Verify your local environment and mask decoder without needing an API key:
```bash
python -m src.api.sam_decoder
```

To re-run dataset manifest inspection or benchmark evaluations:
```bash
python scripts/inspect_carinthia_local.py
python scripts/run_gate0_api_tests.py --mode benchmark20
```
*(Note: Generated PNG overlays and benchmark crops are gitignored and can be regenerated on demand).*
