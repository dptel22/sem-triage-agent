# Compute Plan & Cloud Infrastructure Justification

## 1. Executive Summary for Credit Reviewers (Baseten & Modal)
This document provides a transparent, empirically measured compute audit for the SEM Triage Agent project.

**Key Finding:** Standard convolutional baseline training (ResNet18 U-Net) has been completed entirely on local workstation hardware ($105.65\text{s}$ total runtime, zero cloud compute required). **Cloud credits are requested exclusively for hosted inference and evaluation of open-weight Meta models (`meta-models/muse-spark-open-1026`)** to benchmark the autonomous triage agent against closed API baselines (`muse-spark-1.3`).

---

## 2. Compute Audit Across Pipeline Stages

| Stage | Model | Where It Runs Now | Measured Latency / VRAM | Credits Needed? | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **U-Net Proposer Training** | ResNet18 U-Net | Local workstation (NVIDIA RTX 4050 Laptop GPU) | 105.65s total across 3,618 images (5,026 MiB peak VRAM) | **NO (0 credits)** | **DONE** |
| **U-Net Multi-Split Inference** | ResNet18 U-Net | Local workstation (RTX 4050) | ~2.5s across all 4,591 samples | **NO (0 credits)** | **DONE** |
| **SAM 3.1 Prototyping (Gate 0)** | SAM 3.1 | Meta Model API (`https://api.meta.ai/v1`) | ~1.2s per call (80 calls benchmark, 68 calls forensics) | **NO** (API tier) | **DONE** |
| **Agent Triage Node (API Baseline)** | Muse Spark 1.3 | Meta Model API (`https://api.meta.ai/v1`) | 2,230 tokens/call (including 1,168 reasoning tokens) | **NO** (API tier) | **IN PROGRESS** |
| **Hosted Open-Weights Triage Node** | `meta-models/muse-spark-open-1026` | Not yet deployed (Baseten / Modal target) | Target latency: < 2.5s / query on A10G / L4 | **YES** (Baseten / Modal) | **PLANNED** |
| **Full Frozen Evaluation (480 images)** | `meta-models/muse-spark-open-1026` | Not yet deployed | 430 `nominal_test` + 50 `rare_eval_test` images | **YES** (Baseten / Modal) | **PLANNED** |
| **Multi-Turn Refinement Ablations** | `meta-models/muse-spark-open-1026` | Not yet deployed | ~1,500 total agent inference turns | **YES** (Baseten / Modal) | **PLANNED** |

---

## 3. Measured Local Hardware Benchmarks

All values reported below are physically measured from local execution logs:

```
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, 6,141 MiB VRAM)
Compute Environment: Windows 11 / CUDA 12.1 / PyTorch 2.5.1+cu121
Training Set Size: 3,618 images (nominal_train: 3,437 + class6_train: 181)
Resolution: 480x480 grayscale (border artifact x >= 477 zeroed)
Batch Size: 16 (Mixed Precision AMP enabled)
Measured Training Duration: 105.65 seconds (1.76 minutes across 15 epochs)
Peak VRAM Allocated: 5,026 MiB / 6,141 MiB (81.8% capacity)
Measured Power Draw: 61W – 65W (70W hardware limit)
GPU Utilization: 88% – 100% sustained
Best nominal_val Mean Dice: 0.9098
Cloud Training Credits Required: ZERO ($0.00)
```

---

## 4. Hosted Inference Requirement (`meta-models/muse-spark-open-1026`)

### 4.1 Target Open Model & Hosting Architecture
- **Target Model:** `meta-models/muse-spark-open-1026` (open-weight variant of Muse Spark)
- **Model Parameters / Size:** *Unverified* (official model weights and architecture card pending release/access).
- **Target Hosting Platform:** Serverless GPU container endpoint on **Baseten** or **Modal** with NVIDIA A10G (24 GB VRAM) or L4 (24 GB VRAM).
- **Core Motivation:** Industrial semiconductor fabs operate behind strict air-gapped firewalls. Closed cloud APIs are legally inadmissible for proprietary wafer inspection data. Demonstrating that an open-weight model (`muse-spark-open-1026`) can execute the triage decision node self-hosted inside a fab network is essential for real-world viability.

### 4.2 Workload & Token Throughput Estimation
Token throughput is projected directly from our measured Gate 0 smoke test call (`results/gate0/smoke_result.json`):
- **Input Tokens per Call:** ~1,062 tokens (query crop metadata, Top-5 exemplar distances, morphology table).
- **Output Tokens per Call:** ~1,168 reasoning tokens + ~250 tool-calling tokens (`record_defect_triage` schema).
- **Total Measured Throughput:** **2,230 tokens per triage call**.

### 4.3 Evaluation Budget
1. **Dev Verification & Mechanism Check:** 4 dev samples + 10 frozen test samples $\times$ 2 passes = ~28 calls ($\approx 0.1\text{ GPU-hours}$).
2. **Frozen Full Evaluation:** 480 unseen test images (430 `nominal_test` + 50 `rare_eval_test`) = 480 calls ($\approx 1.5\text{ GPU-hours}$).
3. **Multi-Turn Refinement & Triage Ablations:**
   - Ablation 1: Zero-shot (no retrieval exemplars).
   - Ablation 2: Static rule heuristic (Baseline 1 & 2).
   - Ablation 3: Multi-turn prompt refinement retry loop.
   - Total: ~1,500 inference calls ($\approx 4.5\text{ GPU-hours}$).
4. **Total Compute Request:** $\le 10\text{ GPU-hours}$ on A10G / L4.

---

## 5. Verification Checklist Upon Gaining Access
1. **Multimodal Visual Inputs:** Confirm whether native base64 SEM crops can be passed directly into vision tokenizers or if external patch embeddings are required.
2. **Structured Tool-Calling:** Verify tool-calling schema compatibility with OpenAI function-calling format (`tools=[{"type": "function", ...}]`) vs regex structured output decoding.
3. **Inference Latency:** Measure per-sample latency (target: $< 2.5\text{s}$ per wafer inspection query) to assess fab throughput viability.

---

## 6. Contingency Pivot Plan
*If the project pivots (e.g. Condition C2 fails during agent triage, indicating LLMs cannot reliably separate rare from nominal defects without excessive false alarms):*
Requested cloud compute credits will pivot immediately to training a lightweight embedding metric learner (Supervised Contrastive Learning on ResNet50) to optimize feature distance directly on the semiconductor wafer patches.
