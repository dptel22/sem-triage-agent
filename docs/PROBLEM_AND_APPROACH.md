# Problem & Approach: Semiconductor Defect Triage with Selective Abstention

## 1. Semiconductor Inspection Context & Motivation
Scanning Electron Microscope (SEM) inspection is the ground truth for nano-scale semiconductor fabrication quality control. Industrial Automatic Defect Classification ([ADC](https://electronicimaging.spiedigitallibrary.org/conference-proceedings-of-spie/10145/1014523/Scanning-electron-microscope-automatic-defect-classification-of-process-induced-defects/10.1117/12.2258122.full)) tools—such as [Applied Materials SEMVision](https://www.appliedmaterials.com/eu/en/product-library/g7-semvision-defect-analysis.html)—inspect wafers across production layers to detect physical defects (breaks, voids, particle contamination, structural bridge faults).

However, real-world fab inspection data exhibits extreme class imbalance:
- **Nominal defects** (`class_3`, `class_4`) represent over 93% of all observed defects. Automated convolutional detectors (like U-Net) segment these reliably (median IoU > 0.88).
- **Rare out-of-distribution defects** (`class_1`, `class_2`, `class_5`) represent under 1.5% of total wafer scans (67 out of 4,591 samples in the Carinthia-S benchmark). Standard detectors degrade significantly on these unseen morphologies (~0.66 median IoU).
- **Tool misalignment scans** (`class_6`): Due to SEM beam drift or mechanical alignment jitter, the microscope frequently images clean wafer regions adjacent to the defect, producing empty scans.

In a high-throughput semiconductor fab, a false positive stops the lithography line, costing thousands of dollars per minute in lost wafer throughput. Conversely, an undetected rare defect can indicate a systematic chemical-mechanical polishing (CMP) or etching drift that destroys an entire lot.

---

## 2. Why Triage with Abstention?
Standard computer vision deployments attempt to force a segmentation mask on every input image. When faced with an out-of-distribution rare defect or an empty scan, deep networks output uncalibrated, low-fidelity predictions.

Following selective classification and risk-controlled restoration principles ([SafeRestore, arXiv:2609.03475](https://arxiv.org/abs/2609.03475)), our approach does not attempt to force automated segmentation on impossible targets. Instead, we formulate the problem as **Selective Triage with Abstention**:
1. **Automated Accept:** High-confidence, in-distribution nominal defects (`class_3`, `class_4`) are segmented rapidly by foundation vision models (SAM 3.1) prompted by U-Net bounding boxes.
2. **Selective Deferral (Abstention):** Out-of-distribution rare morphologies (`class_1`, `class_2`, `class_5`) or low-confidence edge cases are flagged and deferred to expert human fab engineers for manual disposition.
3. **Misalignment Routing:** Clean wafer scans (`class_6`) are routed to `"NO_DEFECT / REIMAGE_REQUIRED"` without hallucinating defect polygons.

---

## 3. Empirical Failure Modes Identified (Gate 0 Forensics)

Through empirical benchmarking on the Meta Model API (68 live diagnostic calls recorded in `results/gate0/diagnostic_jitter_and_cleaned_boxes.json`), we uncovered two critical failure modes in foundation model segmentation on SEM imagery:

### 3.1 The Stray-Pixel Bounding Box Artifact (Corner Bug)
- **Forensic Discovery:** In initial zero-shot SAM 3.1 probes, 5 out of 20 images failed catastrophically (IoU < 0.04). Forensics revealed that 3 of these failures (`eb817031`, `0ced545c`, `811a6499`) were caused by 1 to 19 isolated stray noise pixels in the 3-pixel right-hand black border (`x >= 477`) documented in the Carinthia-S dataset specification.
- **Mechanism:** Tight oracle bounding boxes stretched from a compact defect blob (e.g. 13x19 px) all the way to `x=479`. Prompted with this bloated box, SAM 3.1 segmented 30–50% of the wafer frame.
- **Resolution:** Preprocessing must strictly zero the right border artifact (`x >= 477`) and extract connected components with a minimum area threshold ($\ge 16\text{ px}$). Once border pixels were cleaned, SAM 3.1 median IoU jumped from 0.00 to 0.708–0.784.

### 3.2 Thin-Defect Fragility & Multi-Fragment Degradation
- **Forensic Discovery:** Thin line defects (`2326638f`, 22 px wide full-height stripe) and multi-fragment scattered defects (`a957d7c6`, 10 disjoint fragments) cannot be captured by a single bounding box. SAM 3.1 locks onto adjacent wafer texture rather than the localized defect.
- **Resolution:** Rather than expecting single-prompt mask accuracy, these multi-component defects provide an objective trigger for the multimodal agent to either (a) issue iterative point prompts or (b) defer to human inspection.

---

## 4. Formal Research Hypotheses (H1–H3)

### Hypothesis H1: Detector-Prompted Foundation Segmentation Fidelity
* *Hypothesis:* SAM 3.1 prompted by non-oracle, connected-component filtered bounding boxes from a lightweight ResNet18 U-Net achieves $\ge 0.70$ median IoU on nominal semiconductor defects, proving zero-shot foundation models are viable on SEM metrology without retraining.
* *Testing Plan:* Evaluated on all 430 unseen images in `nominal_test` using boxes exported in `results/gate1/unet_predicted_boxes.json`.

### Hypothesis H2: Tool Misalignment Rejection without Hallucination
* *Hypothesis:* Combining U-Net zero-foreground prediction with mask-to-box area verification allows the system to reject $\ge 90\%$ of Class 6 empty wafer scans with 0 fabricated defect masks.
* *Testing Plan:* Evaluated on the 23 empty wafer scans in `class6_test`.

### Hypothesis H3: Exemplar-Guided Triage with Selective Abstention
* *Hypothesis:* A multimodal LLM agent (Muse Spark) utilizing proxy verification metrics (fill ratio, U-Net/SAM mask discordance) and $k$-NN exemplar retrieval achieves higher triage precision and lower false-alarm rates than deterministic heuristic baselines on unseen rare defects.
* *Testing Plan:* Evaluated on the pre-registered 10-sample verification suite (`sharpened-test-frozen`) and extended to all 50 rare evaluation images (`rare_eval_test`) under natural fab prevalence (1.5%).

---

## 5. Non-LLM Comparison Baselines (Controls)
To guarantee that the multimodal LLM agent provides genuine architectural utility rather than superficial complexity, all agent triage decisions are benchmarked against three non-LLM controls evaluated on the identical test cohort:

1. **Baseline 1 (U-Net Maximum Probability Cutoff):**
   - Policy: Defer to human review if U-Net max foreground probability $\max(P_{\text{fg}}) < \tau$ (calibrated on `nominal_val`).
2. **Baseline 2 (U-Net / SAM Mask-Box Discordance):**
   - Policy: Defer if $\text{IoU}(\text{U-Net}, \text{SAM}) < 0.60$ or $\frac{\text{Area}(\text{Mask})}{\text{Area}(\text{Box})} < 0.15$ or $> 0.85$.
3. **Baseline 3 (Non-LLM $k$-NN / Logistic Regression Classifier):**
   - Policy: A deterministic supervised classifier trained on the exact same Top-5 nearest-neighbor distances and quantitative morphological descriptors (area, aspect ratio, fill ratio, solidity).
   - If the LLM agent cannot beat this simple distance-weighted $k$-NN classifier in precision-recall balance, the agent layer is rejected.
