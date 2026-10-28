# SEM Triage Agent: Verifiable Multimodal Wafer Inspection & Selective Abstention

> A verifiable multimodal triage framework evaluating foundation vision (`sam-3.1`) and language models (`muse-spark-1.3` / `muse-spark-open-1026`) on semiconductor SEM defects from the Carinthia-S benchmark ([DOI: 10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427)).

## Problem & Motivation
In semiconductor manufacturing, Automatic Defect Classification ([ADC](https://electronicimaging.spiedigitallibrary.org/conference-proceedings-of-spie/10145/1014523/Scanning-electron-microscope-automatic-defect-classification-of-process-induced-defects/10.1117/12.2258122.full)) systems like [Applied Materials SEMVision](https://www.appliedmaterials.com/eu/en/product-library/g7-semvision-defect-analysis.html) face extreme class imbalance: nominal patterns dominate (>93%), while rare yield-killing anomalies represent under 1.5% of wafer scans. U-Net detectors degrade on rare morphology (~0.66 median IoU), but hallucinating defects on empty tool-misalignment scans triggers costly false fab stops. Following selective risk control ([SafeRestore, arXiv:2609.03475](https://arxiv.org/abs/2609.03475)), this agent automatically segments nominal defects, flags rare anomalies for expert human triage, and routes empty scans to re-imaging without hallucinated masks.

## Status

| Milestone / Capability | Status | Evidence / Artifact |
| :--- | :---: | :--- |
| **Gate 0: SAM 3.1 Zero-Shot Probing & Decoder** | **DONE** | 80 live API calls on real endpoints; bit-exact decoder; `results/gate0/` |
| **Gate 0 Forensics: Jitter & Corner Artifacts** | **DONE** | 68 live calls; 4 of 6 corner failures rescued; `results/gate0/` |
| **Gate 1: Zero-Leakage Dataset Split (8 Splits)** | **DONE** | 4,591 samples verified; `rare_dev` (17) vs `rare_eval_test` (50); `results/gate1/` |
| **Gate 1: ResNet18 U-Net Baseline** | **DONE** | 15 epochs on local RTX 4050 (105.6s); non-oracle boxes exported; `results/gate1/` |
| **Gate 2: Sharpened 10-Sample Verification** | **DONE** | C1, C2, C3 pass; U-Net boxes -> SAM median IoU 0.8615; `results/gate1/` |
| **Gate 3: Natural Refinement & Scaled Evaluation** | **IN PROGRESS** | Negative-point carving & multi-box union; hosted open-weights evaluation |

## Pipeline Architecture

```mermaid
flowchart LR
    Scan["SEM Image"] --> UNet["U-Net Proposer"]
    UNet --> Gate{"Empty-Scan Gate<br/>(Pixels == 0?)"}
    Gate -->|Yes| Reimage["Route: No Defect / Re-Image"]
    Gate -->|No| SAM["SAM 3.1 Proposer (Non-Oracle Box)"]
    SAM --> Verifier["Deterministic Verifier (IoU & Fill)"]
    Verifier --> Retrieve["Top-5 Overall Exemplar Retrieval"]
    Retrieve --> Muse["Muse Spark Decision (1.3 vs Open-1026)"]
    Muse --> Action{"Triage Action"}
    Action -->|High Agreement| Accept["Accept Nominal Mask"]
    Action -->|Discordant / Low Conf| Retry["Refine Prompt & Retry"]
    Action -->|Rare / Out-of-Distribution| Defer["Defer to Human Review"]
```

## Empirical Benchmark Results (Generated from Tracked JSON)

### Gate 0: SAM 3.1 Zero-Shot Benchmark (20 Images, Oracle BBoxes)
| Arm | Scored | Mean IoU | Median IoU | Nominal (3,4,6) Median IoU | Rare (1,2,5) Median IoU | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `text_full` | 0 / 20 | N/A | N/A | N/A | N/A | N/A |
| `box_full` | 20 / 20 | 0.6374 | 0.8572 | 0.8642 | 0.3947 | 0.6881 |
| `crop_roi` | 0 / 20 | N/A | N/A | N/A | N/A | N/A |
| `crop_clahe` | 0 / 20 | N/A | N/A | N/A | N/A | N/A |

### Gate 1: U-Net Baseline Empirical Multi-Split Evaluation (Unseen Splits)
| Split / Cohort | Images (n) | Primary Median IoU (>=128) | Primary Mean IoU | Secondary Median IoU (>0) | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nominal Test (Overall)** | 430 | **0.8869** | 0.8536 | 0.8869 | 0.9170 |
| - `class_3` | 401 | 0.8843 | 0.8482 | 0.8843 | 0.9137 |
| - `class_4` | 29 | 0.9418 | 0.9278 | 0.9418 | 0.9621 |
| **Rare Holdout (Overall)** | 67 | **0.6583** | 0.5981 | 0.6595 | 0.7237 |
| - `class_1` | 55 | 0.6315 | 0.5776 | 0.6283 | 0.7050 |
| - `class_2` | 8 | 0.6046 | 0.6125 | 0.6455 | 0.7553 |
| - `class_5` | 4 | 0.8532 | 0.8508 | 0.8532 | 0.9179 |
| **Class 6 Test (Empty Scans)** | 23 | Raw Spec: **47.83%** (11/23) | Filtered (>=16px) Spec: **82.61% (19/23)** | Mean FP: 6.7 px | Max FP: 37 px |

### Gate 2: Verifiable Multimodal Triage Agent (Official 10-Sample Suite)
| Condition | Protocol Requirement | Empirical Result | Status |
| :--- | :--- | :---: | :---: |
| **C1: Self-Correction Recovery** | $\Delta\text{IoU}_{\text{GT}} \ge +0.20$ (2 candidate cases) and $\Delta \ge -0.05$ (monotonicity) | **+0.8005 and +0.8008** (2/2) | **PASS** |
| **C2: Selective Deferral** | Rare Recall $\ge 66.7\%$ (2/3); Nominal False Alarms $\le 16.7\%$ (1/6) | Rare Recall: **66.67%** (2/3); False Alarms: **0.00%** (0/6) | **PASS** |
| **C3: Empty-Scan Safety** | Route class 6 empty scan to `NO_DEFECT_REIMAGE` with 0 SAM calls | **100.00%** (1/1 test, 23/23 val); 0 paid calls | **PASS** |

## Limitations
- **Single Layer & Unstructured:** Evaluated on a single production layer of unstructured semiconductor wafers.
- **Pixel Units Only:** No physical metrology scale ($\text{nm}$/$\mu\text{m}$) is available in the public dataset.
- **Non-Oracle Bounding Boxes:** U-Net predicted boxes fed to `sam-3.1` achieved 0.8615 median IoU across unseen nominal test images, eliminating oracle-box dependency.
- **Perturbation Recovery vs. Natural Refinement:** Gate 2 C1 recoveries (+0.8005 & +0.8008) tested pipeline fault-recovery from an induced box shift by re-centering on the unperturbed U-Net core box. Natural refinement on unperturbed low-IoU cases (`20cc9526`, Δ = -0.0003) motivates Gate 3 negative-point carving and multi-box queries.
- **Small-Sample Point Estimate ($N=6$):** 0/6 nominal false alarms yields 1.0000 natural precision point estimate, but $N=6$ is too small to bound precision at 1.5% fab prevalence; scaled evaluation ($N \ge 100$) is required.
- **Existing Art:** Commercial rule-based ADC systems already operate in production fabs; this work evaluates multimodal foundation models with abstention.

## Quickstart (6 Tested Commands)
```bash
git clone https://github.com/sem-triage/sem-triage-agent.git && cd sem-triage-agent
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -v
python scripts/make_results_table.py --print
python scripts/select_sharpened_test_samples.py
```

## Documentation & Compliance Links
- `docs/PROBLEM_AND_APPROACH.md`: Semiconductor triage background, abstention, and hypotheses H1-H3.
- `docs/COMPUTE_PLAN.md`: Measured local compute benchmarks (zero training credits needed) and hosted inference plan for `muse-spark-open-1026`.
- `docs/HACKATHON_COMPLIANCE.md`: Rules compliance, component lineage, CC BY 4.0 rights.
- `docs/sharpened_test_protocol.md`: Pre-registered frozen mechanism verification protocol.

## Data Attribution & License
- **Dataset:** Carinthia-S ([DOI: 10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427)), by Corinna Kofler (KAI GmbH), Vahidin Hasić (University of Sarajevo), et al., licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original Carinthia benchmark: [DOI: 10.5281/zenodo.10715190](https://doi.org/10.5281/zenodo.10715190).
- **Code License:** [MIT License](LICENSE).
