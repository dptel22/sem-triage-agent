# Gate 0 Benchmark & Validation Summary

Empirical validation results for `sam-3.1` and `muse-spark-1.3` on SEM wafer-defect imagery via the Meta Model API (`https://api.meta.ai/v1`).

---

## 1. Scope & Execution

- **Total API Calls Tested**:
  - Smoke test: 4 probes on synthetic fixture image + positive-control validation on natural photos.
  - 20-image benchmark: 80 total live API calls (20 images across 4 prompt arms: `text_full`, `box_full`, `crop_roi`, `crop_clahe`).
  - All calls executed against real endpoints with zero fabricated numbers or mocks.

---

## 2. Benchmark Results per Arm (sam-3.1, 20 Images)

Data sourced directly from `results/gate0/sam_20_results.json`:

| Arm | Scored Images | Overall Mean IoU | Overall Median IoU | Nominal Classes (3, 4, 6) Median IoU | Rare Classes (1, 2, 5) Median IoU | Mean Dice |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `text_full` (full-image text prompt: "defect") | 0 / 20 | N/A | N/A | N/A | N/A | N/A |
| `box_full` (full-image tight box prompt) | 20 / 20 | 0.6374 | 0.8572 | 0.8642 | 0.3947 | 0.6881 |
| `crop_roi` (padded bounding box crop + text) | 0 / 20 | N/A | N/A | N/A | N/A | N/A |
| `crop_clahe` (CLAHE-enhanced crop + text) | 0 / 20 | N/A | N/A | N/A | N/A | N/A |

### Observations:
- In `text_full`, `crop_roi`, and `crop_clahe`, all 60 API calls returned HTTP 200 with `status=completed` and empty `output_text` (zero matches detected). CLAHE contrast enhancement did not resolve text detection on SEM imagery.
- In `box_full`, SAM accepted all 20 prompts and produced masks. 14 out of 20 images achieved strong segmentation (median IoU > 0.85).

---

## 3. Factual Evidence from Gate 0 Probes

- **Positive Control Text Prompt on Natural Photo**:
  To confirm the text prompting pipeline was mechanically operational, a control test was conducted on a natural photo (`truck.jpg`, 1800x1200) with prompt `"truck"`. The API returned a 68,770-character response containing a 356x1024 lossless mask that decoded bit-exactly. A cropped ROI test on the same image returned a 119,756-character response with a 900x518 mask. This confirms that empty text responses on SEM defect images reflect the model's domain-specific zero-detection behavior, not an API or harness defect.
- **Muse Spark 1.3 Tool Calling**:
  Tested in `smoke_result.json` on a fixture image with `tool_choice="auto"`. Returned `finish_reason="tool_calls"` calling `record_defect_triage` with valid structured arguments (`action="review"`, `confidence=0.65`, `morphology_type="scratch"`) alongside reasoning prose accurately describing the defect features.

---

## 4. Mandatory Caveats

1. **Oracle Bounding Boxes**: The prompt boxes in `box_full` were derived directly from ground-truth masks (oracle boxes). Thus, the median IoU of 0.8572 represents an upper bound for SAM-3.1 segmentation fidelity, not an end-to-end detector-driven result.
2. **Edge-Touching Defects (Low IoU Failures)**: 5 images scored IoU < 0.04 (specifically sample indices with prompt boxes touching top/right boundaries, e.g. `y1=0` or `x2=479`). In these instances, SAM segmented adjacent border/wafer edge structures rather than the localized defect. **Root cause resolved by the follow-up diagnostic — see Section 5.**
3. **Class 6 Mask Distribution**: The single Class 6 sample included in the 20-image benchmark contains an atypical non-empty mask; 224 of the 227 Class 6 images in the full Carinthia-S dataset possess completely empty masks.
4. **Tool Calling Sample Size**: Muse Spark 1.3 function calling was evaluated on a single test image during Gate 0 smoke testing.

---

## 5. Gate 0 Follow-Up Diagnostic (Cleaned Boxes & Jitter Sensitivity)

Branch `feat/gate0-diagnostics`; script `scripts/diagnose_failures_and_jitter.py`;
full evidence in `results/gate0/diagnostic_jitter_and_cleaned_boxes.json`
(every call retains `response_id` + `raw_output_text`; SHA-256 of all pre-existing
`results/gate0/*.json` verified identical before and after). 68 live `sam-3.1` calls.

### 5.1 Root cause of the 5 corner-box failures (Part A forensics)

- **3 of 5 failures (`eb817031`, `0ced545c`, `811a6499`) were caused by stray pixels in the 3px right-border artifact, not by soft-mask halo.** Only 1–19 border pixels existed, but they stretched the oracle `tight_bbox_xyxy` from a tiny dense core (e.g. a 13x19 px blob for `eb817031`) across half the image; `sam-3.1` then dutifully segmented the whole boxed region (31–50% of the frame).
- `2326638f` is a genuine 22px-wide, full-height (y=0..478) thin stripe — a ground-truth-semantics mismatch, not strays.
- `a957d7c6` is a 10-fragment scattered defect (components 2814/1535/1074/286/228 px); no single box can cover it.
- The Class 6 exception `8b782473` is a clean 16x8 px, 50-pixel blob — cleaning is the identity; its 0.38 IoU is a small-object precision limit.

**Cleaned-box re-calls (zeroing `x >= 477`, then `box_thr128` / `box_lcc_gt0` / `box_lcc_128`):**

| Image | class | oracle IoU | best cleaned box | cleaned box [x1, y1, x2, y2] | best IoU (gt>0 / gt>=128) |
|---|---|---|---|---|---|
| `a957d7c6` | 1 | 0.034 | `box_lcc_gt0` | `[283, 157, 424, 336]` | 0.309 / 0.312 |
| `2326638f` | 2 | 0.017 | `box_thr128` (all identical) | `[225, 0, 246, 478]` | 0.620 / 0.670 |
| `811a6499` | 2 | 0.014 | `box_thr128` (all identical) | `[0, 138, 476, 346]` | 0.664 / 0.719 |
| `eb817031` | 3 | 0.000 | `box_thr128` (all identical) | `[236, 234, 248, 252]` | 0.784 / 0.785 |
| `0ced545c` | 4 | 0.000 | `box_thr128` (all identical) | `[224, 228, 250, 250]` | 0.708 / 0.745 |
| `8b782473` | 6 | 0.381 | unchanged (identity) | `[203, 269, 218, 276]` | 0.381 / 0.381 |

Border-pixel cleaning rescued 4 of 6 failures into the 0.62–0.78 range; the multi-fragment class-1 defect recovers partially (0.31); the Class 6 small object is unchanged.

### 5.2 Symmetric ±15% per-edge jitter (Part B, 2 trials/image, seed 42)

*Note: Statistics below strictly separate oracle-box and cleaned-box trials (no pooling).*

| Group (prompt box jittered) | Prompt Box Source | n | median IoU (gt>0) | mean IoU (gt>0) | median Dice |
|---|---|---|---|---|---|
| 14 clean images | Oracle box jittered | 28 | **0.8698** | 0.7999 | 0.9304 |
| 5 corner images | Oracle box jittered | 10 | **0.0029** | 0.0337 | 0.0057 |
| 5 corner images | Cleaned `box_lcc_128` jittered | 10 | **0.6020** (0.6547 vs gt>=128) | 0.5874 | 0.7514 |
| Class 6 exception | Oracle box jittered | 2 | 0.2500 | 0.2500 | 0.4000 |
| All 20 images | Oracle box jittered | 40 | **0.8193** | 0.5809 | 0.9007 |

### 5.3 Diagnostic Findings & Threshold Evaluation

1. **Threshold Evaluation**:
   - On the 14 clean images, the jittered median IoU of **0.8698** clears the proposed $\ge 0.70$ threshold. Compared to their unjittered baseline of **0.8750** (clean-14 median, distinct from the all-20 median of 0.8572), performance is essentially flat ($-0.0052$).
   - On the corner images, cleaned boxes lift jittered median IoU to **0.6020** (and **0.6547** vs `gt>=128`), clearing the proposed $\ge 0.50$ threshold. This confirms the corner failures were ground-truth box construction artifacts rather than intrinsic SAM-3.1 model failures.
   - The SAM arm survives for compact defects with high-quality boxes.
2. **Defect Geometry & Thin/Fragmented Fragility**:
   - Compact defects (Classes 3, 4, 5) remain resilient under jitter ($0.85$–$0.94$ IoU).
   - In contrast, thin and multi-fragment defects are fragile: Class 1 sample `1079519b` dropped from $0.7551$ unjittered to $0.0321$ and $0.2964$ under jitter. Similarly, cleaned boxes for Class 1 multi-fragment sample `a957d7c6` only reached $\sim 0.309$. Given that rare-class handling (such as Class 1) is central to the project, thin/fragmented defect segmentation represents a key operational risk.
3. **Caveat on "Robustness"**:
   - Synthetic $\pm 15\%$ per-edge jitter around ground-truth boxes tests sensitivity to bounding box margins, but does **not** represent true detector error (which includes false positives, aspect ratio distortion, and center offset). End-to-end viability must be verified using actual predicted bounding boxes from the Gate 1 U-Net baseline.
4. **Immediate Actionable Pipeline Adjustments**:
   - Zero out the 3px right border (`x >= 477`) before computing any tight bbox from masks or detections.
   - Prefer largest-connected-component (`box_lcc_128`) bounding box extraction to avoid spanning stray noise pixels.
