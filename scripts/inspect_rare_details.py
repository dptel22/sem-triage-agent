import json

with open("results/gate1/unet_baseline_results.json", "r") as f:
    d = json.load(f)

per_img = d["rare_holdout_per_image"]
c1_imgs = [x for x in per_img if x["class_label"] == 1]
c2_imgs = [x for x in per_img if x["class_label"] == 2]
c5_imgs = [x for x in per_img if x["class_label"] == 5]

print(f"Class 1 (ridges, n={len(c1_imgs)}):")
print(f"  IoU Primary: Median={d['rare_holdout_summary']['per_class']['class_1']['median_iou_primary']:.4f}, Mean={d['rare_holdout_summary']['per_class']['class_1']['mean_iou_primary']:.4f}")
print("  Sample per-image entries (first 5 of 55):")
for x in c1_imgs[:5]:
    print(f"    id={x['id']} | iou_pri={x['iou_primary']:.4f} | iou_sec={x['iou_secondary']:.4f} | pred_px={x['predicted_pixels']} | gt_px={x['ground_truth_primary_pixels']} | gate0={x['seen_in_gate0']}")

print(f"\nClass 2 (n={len(c2_imgs)}):")
print(f"  IoU Primary: Median={d['rare_holdout_summary']['per_class']['class_2']['median_iou_primary']:.4f}, Mean={d['rare_holdout_summary']['per_class']['class_2']['mean_iou_primary']:.4f}")
print("  All 8 per-image entries:")
for x in c2_imgs:
    print(f"    id={x['id']} | iou_pri={x['iou_primary']:.4f} | iou_sec={x['iou_secondary']:.4f} | pred_px={x['predicted_pixels']} | gt_px={x['ground_truth_primary_pixels']} | gate0={x['seen_in_gate0']}")

print(f"\nClass 5 (n={len(c5_imgs)}):")
print(f"  IoU Primary: Median={d['rare_holdout_summary']['per_class']['class_5']['median_iou_primary']:.4f}, Mean={d['rare_holdout_summary']['per_class']['class_5']['mean_iou_primary']:.4f}")
print("  All 4 per-image entries:")
for x in c5_imgs:
    print(f"    id={x['id']} | iou_pri={x['iou_primary']:.4f} | iou_sec={x['iou_secondary']:.4f} | pred_px={x['predicted_pixels']} | gt_px={x['ground_truth_primary_pixels']} | gate0={x['seen_in_gate0']}")
