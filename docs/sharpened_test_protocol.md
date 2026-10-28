# Sharpened 10-Image Verification Protocol: Triage Agent Mechanism Proof

## 1. Executive Summary & Problem Formulation
The original test proposal suffered from four fatal methodological vulnerabilities:
1. **Cherry-Picking via Weak Condition ("At least one image"):** Passing on a single image can occur by pure stochastic variance or chance prompt alignment.
2. **Trivial Deferral Exploit ("Flag everything"):** An agent that indiscriminately defers on all inputs achieves 100% rare recall while destroying operational utility. False alarm rate on nominal defects must be bounded.
3. **Subjective Self-Assessment:** Asking the model whether a segmentation is "good" or "bad" leads to sycophantic self-justification. Refinement and triage triggers must rely on observable proxy metrics (e.g., mask-box fill ratio, U-Net/SAM mask disagreement, boundary contact) and be scored against withheld ground truth that the agent never observes.
4. **Absence of a Non-LLM Baseline:** If a simple threshold rule (e.g., defer when U-Net confidence $< 0.85$ or U-Net/SAM IoU $< 0.50$) matches agent triage accuracy, the LLM adds zero proven architectural value.

This protocol pre-registers exact frozen sample IDs, quantitative pass/fail thresholds, baseline controls, and an immutable decision tree *prior* to test execution.

---

## 2. Pre-Registered 10-Sample Cohort (Seed 42)
Generated deterministically from `results/gate1/splits_manifest.json` via seed 42. No samples were seen during Gate 0 API tests.

| Role | Image ID | Class | Class Name | Ground-Truth State | Split Source |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `rare_c1_unseen` | `d1147ea97bb0410487f432a49def8265` | 1 | Particle / Dust | Non-empty (Rare) | `rare_holdout` |
| `rare_c2_unseen` | `20cc9526c58743019f12544ca3f6ffee` | 2 | Bridge / Short | Non-empty (Rare) | `rare_holdout` |
| `rare_c5_unseen` | `75e64833559e418b8ee5a3760b0b6f21` | 5 | Delamination / Blister | Non-empty (Rare) | `rare_holdout` |
| `empty_scan_c6` | `404e77708f1649dcb94ce4e692741ed4` | 6 | SEM Misalignment | Empty (False Alarm) | `class6_test` |
| `nominal_c3_1` | `52b03c7a76c340cfb4cb40d2685673f1` | 3 | Break / Open | Non-empty (Nominal) | `nominal_test` |
| `nominal_c3_2` | `4e27428d290e4c7ebbccae1c93920343` | 3 | Break / Open | Non-empty (Nominal) | `nominal_test` |
| `nominal_c3_3` | `2dade3ad5e1b4ed3a1c43952dd5c9b6c` | 3 | Break / Open | Non-empty (Nominal) | `nominal_test` |
| `nominal_c3_4` | `ef3a94a2231d410aa67eeb24f7765f19` | 3 | Break / Open | Non-empty (Nominal) | `nominal_test` |
| `nominal_c4_1` | `1bf20f9c19b642438dc547aa16f09207` | 4 | Via Void / Missing Contact | Non-empty (Nominal) | `nominal_test` |
| `nominal_c4_2` | `bdc04b42c49c4ceab2543b56b2a441af` | 4 | Via Void / Missing Contact | Non-empty (Nominal) | `nominal_test` |

### Blinding Protocol:
The test harness passes images to the agent stripped of:
- `class_label`
- `class_name`
- `ground_truth_mask`
- `ground_truth_bbox`
Exemplar library queries (retrieval) provide labeled historical examples, but the target test image remains strictly unlabelled.

---

## 3. Quantitative Conditions & Pre-Registered Thresholds

### Condition C1: Objective Self-Correction & Refinement
*Objective:* Verify that the agent can detect a flawed first-pass segmentation and improve it without degrading already accurate segmentations.

- **Test Sets:**
  - *Natural low-IoU cases:* Any test sample where first-pass IoU $< 0.50$.
  - *Induced perturbation test:* To ensure reproducible low-quality first passes, exactly 2 nominal test images (`nominal_c3_1` and `nominal_c4_1`) receive an induced degraded prompt box (expanded by $+25\%$ and shifted by $+15$ pixels, creating intentional background leakage).
- **Pass Thresholds:**
  1. **Recovery Delta:** Across the candidate low-pass/induced images (minimum 2), the agent must trigger a refinement retry on at least 2 instances and achieve:
     $$\Delta \text{IoU} = \text{IoU}_{\text{refined}} - \text{IoU}_{\text{initial}} \ge +0.20$$
  2. **Monotonicity / Safety Guardrail:** On images where initial $\text{IoU} \ge 0.80$, the agent must NOT degrade mask quality by more than $0.05$:
     $$\Delta \text{IoU} \ge -0.05$$

---

### Condition C2: Selective Deferral vs Deterministic Baseline
*Objective:* Verify that the agent discriminates between out-of-distribution rare defects (requiring human engineer review) and nominal in-distribution defects, outperforming standard heuristic rules.

- **Agent Thresholds:**
  - **Rare Defect Sensitivity (Recall):** Flag $\ge 2 / 3$ ($66.7\%$) of unseen rare images (`rare_c1`, `rare_c2`, `rare_c5`).
  - **Nominal Specificity (False Alarm Rate):** Falsely flag $\le 1 / 6$ ($16.7\%$) nominal images (hard ceiling: $\le 2 / 6$, $33.3\%$).
- **Control Baseline Comparison:**
  The identical 10 images are evaluated against two deterministic rule-based baselines:
  - *Baseline A (Confidence):* Defer if U-Net max foreground probability $< 0.85$.
  - *Baseline B (Discordance):* Defer if $\text{IoU}(\text{U-Net}, \text{SAM}) < 0.60$ or $\frac{\text{Area}(\text{Mask})}{\text{Area}(\text{Box})} < 0.15$.
  - *Pass Criterion:* Agent Triage F1-score must exceed both Baseline A and Baseline B.

---

### Condition C3: Class 6 Semantic Routing
*Objective:* Verify robust handling of tool misalignment / empty wafer scans.

- **Sample:** `empty_scan_c6` (`404e77708f1649dcb94ce4e692741ed4`).
- **Pass Thresholds:**
  1. The agent routes the image to `"NO_DEFECT / REIMAGE_REQUIRED"`.
  2. Zero defect polygons exported to downstream CAD/yield database.
  3. No hallucinated bounding box submitted to SAM.

---

## 4. Execution Integrity & Audit Trail
1. **Single-Run Policy:** The verification suite is executed exactly once against the 10 frozen images. No hyperparameter tuning or prompt mutation is permitted after results are inspected.
2. **Audit Manifest:** All raw API responses, SAM tokens, intermediate prompt boxes, U-Net confidence maps, and execution timestamps are recorded in `results/gate1/sharpened_test_results.json` (<1 MB).
3. **Harness Ground Truth Isolation:** Ground truth scoring is performed exclusively inside the test harness script. The agent never receives or inspects ground truth masks.

---

## 5. Architectural Decision Matrix

| Condition C1 (Self-Correction) | Condition C2 (Selective Deferral) | Decision Outcome | Strategic Next Step |
| :---: | :---: | :---: | :--- |
| **PASS** | **PASS** | **LOCK PROJECT** | Proceed immediately to Gate 2 full evaluation (67 rare + 430 nominal). Architecture fully validated. |
| **FAIL** | **PASS** | **LOCK CAUTIOUSLY** | Deferral mechanism works. Eliminate iterative interactive prompt loops from critical path; lock agent as a high-precision Triage Router. |
| **PASS** | **FAIL** | **PIVOT** | Deferral is the core semiconductor triage differentiator. If it cannot reliably flag rare defects, the LLM agent is redundant. |
| **FAIL** | **FAIL** | **PIVOT / TERMINATE** | Both interactive refinement and deferral failed. Revert to standalone U-Net baseline + static rules. |
