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
2. **Edge-Touching Defects (Low IoU Failures)**: 5 images scored IoU < 0.04 (specifically sample indices with prompt boxes touching top/right boundaries, e.g. `y1=0` or `x2=479`). In these instances, SAM segmented adjacent border/wafer edge structures rather than the localized defect. Root cause analysis is ongoing.
3. **Class 6 Mask Distribution**: The single Class 6 sample included in the 20-image benchmark contains an atypical non-empty mask; 224 of the 227 Class 6 images in the full Carinthia-S dataset possess completely empty masks.
4. **Tool Calling Sample Size**: Muse Spark 1.3 function calling was evaluated on a single test image during Gate 0 smoke testing.
