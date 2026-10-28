import json

with open("results/gate1/unet_baseline_results.json", "r") as f:
    data = json.load(f)

print("=== NOMINAL TEST SUMMARY ===")
nt = data["nominal_test"]
print(f"Overall Primary Median IoU: {nt['median_iou_primary']:.4f}, Mean IoU: {nt['mean_iou_primary']:.4f}, Mean Dice: {nt['mean_dice_primary']:.4f}")
print(f"Overall Secondary Median IoU: {nt['median_iou_secondary']:.4f}, Mean IoU: {nt['mean_iou_secondary']:.4f}, Mean Dice: {nt['mean_dice_secondary']:.4f}")

for cname, stats in nt["per_class"].items():
    print(f"  {cname} (n={stats['count']}):")
    print(f"    Primary (mask >= 128):   Median IoU = {stats['median_iou_primary']:.4f}, Mean IoU = {stats['mean_iou_primary']:.4f}, Mean Dice = {stats['mean_dice_primary']:.4f}")
    print(f"    Secondary (mask > 0):    Median IoU = {stats['median_iou_secondary']:.4f}, Mean IoU = {stats['mean_iou_secondary']:.4f}, Mean Dice = {stats['mean_dice_secondary']:.4f}")

print("\n=== CLASS 6 TEST SUMMARY ===")
c6 = data["class6_test"]
esm = c6["empty_scan_metrics"]
print(f"Empty-scan specificity: {esm['empty_scan_specificity']}% ({esm['empty_scans_zero_predicted']}/{esm['empty_scans_total']} empty scans with 0 predicted FG pixels)")
print(f"False positive pixels on empty scans: Mean = {esm['mean_false_positive_pixels']:.2f}, Median = {esm['median_false_positive_pixels']}, Max = {esm['max_false_positive_pixels']}")

print("\n=== RARE HOLDOUT SUMMARY ===")
rh = data["rare_holdout_summary"]
print(f"Overall Primary Median IoU: {rh['median_iou_primary']:.4f}, Mean IoU: {rh['mean_iou_primary']:.4f}, Mean Dice: {rh['mean_dice_primary']:.4f}")
for cname, stats in rh["per_class"].items():
    print(f"  {cname} (n={stats['count']}):")
    print(f"    Primary (mask >= 128):   Median IoU = {stats['median_iou_primary']:.4f}, Mean IoU = {stats['mean_iou_primary']:.4f}, Mean Dice = {stats['mean_dice_primary']:.4f}")
    print(f"    Secondary (mask > 0):    Median IoU = {stats['median_iou_secondary']:.4f}, Mean IoU = {stats['mean_iou_secondary']:.4f}, Mean Dice = {stats['mean_dice_secondary']:.4f}")

rare_samples = data["rare_holdout_per_sample"]
print(f"\nTotal individual rare holdout samples logged: {len(rare_samples)}")
print("First 5 samples:")
for s in rare_samples[:5]:
    print(f"  ID: {s['id']} | Class: {s['label']} | Primary IoU: {s['iou_primary']:.4f} | Sec IoU: {s['iou_secondary']:.4f} | FG Pred: {s['pred_pixels']} | GT: {s['gt_pixels_primary']}")
