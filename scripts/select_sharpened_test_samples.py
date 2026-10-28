import json
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
        # Neutral ground-truth class labels
        class_names = {
            1: "class_1 (rare)",
            2: "class_2 (rare)",
            3: "class_3 (nominal)",
            4: "class_4 (nominal)",
            5: "class_5 (rare)",
            6: "class_6 (no visible defect / SEM misalignment)",
        }
        samples_by_id[sid] = {
            "id": sid,
            "class_label": label,
            "class_name": class_names.get(label, f"class_{label}"),
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

# Rare eval test (unseen in Gate 0)
rare_samples = splits["rare_eval_test"]
rare_c1_unseen = [samples_by_id[s["id"]] for s in rare_samples if s["class_label"] == 1 and s["id"] not in gate0_ids]
rare_c2_unseen = [samples_by_id[s["id"]] for s in rare_samples if s["class_label"] == 2 and s["id"] not in gate0_ids]
rare_c5_unseen = [samples_by_id[s["id"]] for s in rare_samples if s["class_label"] == 5 and s["id"] not in gate0_ids]

# Class 6 empty in class6_test
c6_test = splits["class6_test"]
c6_empty = [samples_by_id[s["id"]] for s in c6_test if s["id"] not in gate0_ids]

# Nominal test
nom_test = splits["nominal_test"]
nom_c3 = [samples_by_id[s["id"]] for s in nom_test if s["class_label"] == 3 and s["id"] not in gate0_ids]
nom_c4 = [samples_by_id[s["id"]] for s in nom_test if s["class_label"] == 4 and s["id"] not in gate0_ids]

# Preserve exact pre-registered sample IDs:
pre_registered_ids = [
    ("rare_c1_unseen", "d1147ea97bb0410487f432a49def8265"),
    ("rare_c2_unseen", "20cc9526c58743019f12544ca3f6ffee"),
    ("rare_c5_unseen", "75e64833559e418b8ee5a3760b0b6f21"),
    ("empty_scan_c6", "404e77708f1649dcb94ce4e692741ed4"),
    ("nominal_c3_1", "52b03c7a76c340cfb4cb40d2685673f1"),
    ("nominal_c3_2", "4e27428d290e4c7ebbccae1c93920343"),
    ("nominal_c3_3", "2dade3ad5e1b4ed3a1c43952dd5c9b6c"),
    ("nominal_c3_4", "ef3a94a2231d410aa67eeb24f7765f19"),
    ("nominal_c4_1", "1bf20f9c19b642438dc547aa16f09207"),
    ("nominal_c4_2", "bdc04b42c49c4ceab2543b56b2a441af"),
]

manifest = []
for role, sid in pre_registered_ids:
    s = samples_by_id[sid]
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
        "manifest_version": "4.0",
        "seed": 42,
        "purpose": "Pre-registered, frozen 10-sample verification suite for Triage Agent mechanism proof (Conditions C1, C2, C3)",
        "composition": {
            "unseen_rare_eval_test": 3,
            "empty_scan_class6": 1,
            "nominal_test": 6
        },
        "taxonomy_note": (
            "Neutral ground-truth class labels: class_1 (rare), class_2 (rare), "
            "class_3 (nominal), class_4 (nominal), class_5 (rare), and "
            "class_6 (no visible defect / SEM misalignment)."
        ),
        "samples": manifest
    }, f, indent=2)

print(f"\nSaved neutral manifest to {out_path} ({out_path.stat().st_size} bytes)")
