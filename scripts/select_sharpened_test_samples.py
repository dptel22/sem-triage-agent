import json
import random
import csv
from pathlib import Path

# Load carinthia-s.csv (semicolon-delimited)
csv_path = Path("data/carinthia_s/carinthia-s.csv")
samples_by_id = {}
with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        filename = row["filename"]
        sid = Path(filename).stem
        label = int(row["label"])
        class_names = {
            1: "Particle / Dust (Rare)",
            2: "Bridge / Short (Rare)",
            3: "Break / Open (Nominal)",
            4: "Via Void / Missing Contact (Nominal)",
            5: "Delamination / Blister (Rare)",
            6: "SEM Misalignment / Empty (False Alarm)",
        }
        samples_by_id[sid] = {
            "id": sid,
            "class_label": label,
            "class_name": class_names.get(label, f"Class_{label}"),
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "filename": filename,
        }

# Load Gate 0 IDs
with open("results/gate0/gate0_20_manifest.json", "r", encoding="utf-8") as f:
    g0 = json.load(f)
gate0_ids = set(s["id"] for s in g0["samples"])

# Load Gate 1 Splits
with open("results/gate1/splits_manifest.json", "r", encoding="utf-8") as f:
    splits_data = json.load(f)
splits = splits_data["splits"]

# Rare unseen in Gate 0
rare_samples = splits["rare_holdout"]
rare_c1_unseen = [samples_by_id[s["id"]] for s in rare_samples if s["class_label"] == 1 and s["id"] not in gate0_ids]
rare_c2_unseen = [samples_by_id[s["id"]] for s in rare_samples if s["class_label"] == 2 and s["id"] not in gate0_ids]
rare_c5_unseen = [samples_by_id[s["id"]] for s in rare_samples if s["class_label"] == 5 and s["id"] not in gate0_ids]

print(f"Rare C1 unseen count: {len(rare_c1_unseen)}/55")
print(f"Rare C2 unseen count: {len(rare_c2_unseen)}/8")
print(f"Rare C5 unseen count: {len(rare_c5_unseen)}/4 (2 seen in Gate 0, exactly 2 remain!)")

# Class 6 empty in class6_test
c6_test = splits["class6_test"]
# In class6_test, check which ones are empty (all class 6 in class6_test are empty, non-empty was routed to class6_train)
c6_empty = [samples_by_id[s["id"]] for s in c6_test if s["id"] not in gate0_ids]
print(f"Class 6 in class6_test: {len(c6_empty)}")

# Nominal test
nom_test = splits["nominal_test"]
nom_c3 = [samples_by_id[s["id"]] for s in nom_test if s["class_label"] == 3 and s["id"] not in gate0_ids]
nom_c4 = [samples_by_id[s["id"]] for s in nom_test if s["class_label"] == 4 and s["id"] not in gate0_ids]
print(f"Nominal C3 in test: {len(nom_c3)}")
print(f"Nominal C4 in test: {len(nom_c4)}")

# Deterministic selection with seed 42
random.seed(42)
sel_c1 = random.choice(rare_c1_unseen)
sel_c2 = random.choice(rare_c2_unseen)
sel_c5 = random.choice(rare_c5_unseen)
sel_c6 = random.choice(c6_empty)
sel_c3 = random.sample(nom_c3, 4)
sel_c4 = random.sample(nom_c4, 2)

selected = [
    ("rare_c1_unseen", sel_c1),
    ("rare_c2_unseen", sel_c2),
    ("rare_c5_unseen", sel_c5),
    ("empty_scan_c6", sel_c6),
    ("nominal_c3_1", sel_c3[0]),
    ("nominal_c3_2", sel_c3[1]),
    ("nominal_c3_3", sel_c3[2]),
    ("nominal_c3_4", sel_c3[3]),
    ("nominal_c4_1", sel_c4[0]),
    ("nominal_c4_2", sel_c4[1]),
]

manifest = []
for role, s in selected:
    print(f"Role: {role:<16} | ID: {s['id']} | Class {s['class_label']} ({s['class_name']})")
    manifest.append({
        "role": role,
        "id": s["id"],
        "class_label": s["class_label"],
        "class_name": s["class_name"],
        "image_path": s["image_path"],
        "mask_path": s["mask_path"],
        "filename": s["filename"],
        "seen_in_gate0": False
    })

out_path = Path("results/gate1/sharpened_test_10_manifest.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "manifest_version": "1.0",
        "seed": 42,
        "purpose": "Pre-registered, frozen 10-sample verification suite for Triage Agent mechanism proof (Conditions C1, C2, C3)",
        "composition": {
            "unseen_rare_holdout": 3,
            "empty_scan_class6": 1,
            "nominal_test": 6
        },
        "samples": manifest
    }, f, indent=2)

print(f"\nSaved pre-registered manifest to {out_path} ({out_path.stat().st_size} bytes)")
