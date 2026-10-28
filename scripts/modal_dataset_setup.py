"""Modal cloud setup for Carinthia-S dataset volume.

Prepares a persistent Modal Volume ('carinthia-s-vol') containing the
Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427, CC BY 4.0,
single production layer of unstructured wafers) for cloud GPU training in Gate 1.
"""

from pathlib import Path
import sys
import modal

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = modal.App("sem-triage-gate0")
volume = modal.Volume.from_name("carinthia-s-vol", create_if_missing=True)

# Container image with requests and tqdm for cloud dataset download
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "requests>=2.31.0",
    "tqdm>=4.66.0"
)

VOLUME_MOUNT_PATH = "/mnt/carinthia_s"
ZENODO_RECORD_URL = "https://zenodo.org/api/records/16895427"


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def download_and_mount_carinthia_on_modal():
    """Downloads Carinthia-S from Zenodo directly into Modal Volume in cloud."""
    import io
    import shutil
    import requests
    import zipfile
    from tqdm import tqdm

    target_dir = Path(VOLUME_MOUNT_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking Zenodo API: {ZENODO_RECORD_URL}")
    resp = requests.get(ZENODO_RECORD_URL, timeout=30)
    resp.raise_for_status()
    rec = resp.json()

    files = rec.get("files", [])
    print(f"Record contains {len(files)} files:")
    for f in files:
        print(f"  - {f['key']} ({f['size']} bytes)")

    # 1. Download HTML metadata
    for f in files:
        if f["key"] == "carinthia-s_dataset.html":
            dest = target_dir / "carinthia-s_dataset.html"
            if not dest.exists() or dest.stat().st_size != f["size"]:
                print(f"Downloading {f['key']}...")
                r = requests.get(f["links"]["self"], timeout=60)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"Saved {dest} ({dest.stat().st_size} bytes)")
            else:
                print(f"{dest.name} already exists on volume.")

    # 2. Download and extract data.zip
    for f in files:
        if f["key"] == "data.zip":
            # Check if extracted files already exist
            csv_path = target_dir / "carinthia-s.csv"
            images_dir = target_dir / "images"
            if csv_path.exists() and images_dir.exists():
                print("Dataset files already extracted on volume.")
            else:
                print(f"Downloading {f['key']} ({f['size']} bytes)...")
                r = requests.get(f["links"]["self"], stream=True, timeout=300)
                r.raise_for_status()
                zip_bytes = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1048576):
                    zip_bytes.write(chunk)
                print("Extracting data.zip to volume...")
                zip_bytes.seek(0)
                with zipfile.ZipFile(zip_bytes) as z:
                    z.extractall(target_dir)

                # Flatten nested data folder if created by zip
                nested_dir = target_dir / "data"
                if nested_dir.exists():
                    for item in nested_dir.iterdir():
                        dest = target_dir / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                    try:
                        nested_dir.rmdir()
                    except Exception:
                        pass
                print("Extraction and flattening complete.")

    # Commit volume changes
    volume.commit()
    print("Modal Volume 'carinthia-s-vol' successfully committed.")

    # Inventory volume contents
    summary = {
        "dataset_attribution": "Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427, CC BY 4.0, single production layer of unstructured wafers)",
        "volume_name": "carinthia-s-vol",
        "mount_path": VOLUME_MOUNT_PATH,
        "files_count": len([p for p in target_dir.rglob("*") if p.is_file()]),
        "has_csv": (target_dir / "carinthia-s.csv").exists(),
        "has_html": (target_dir / "carinthia-s_dataset.html").exists(),
        "images_count": len(list((target_dir / "images").glob("*.jpg"))) if (target_dir / "images").exists() else 0,
        "masks_count": len(list((target_dir / "masks").glob("*.png"))) if (target_dir / "masks").exists() else 0,
    }
    print("Volume verification summary:", summary)
    return summary


@app.local_entrypoint()
def main():
    print("Executing Modal cloud dataset setup for 'carinthia-s-vol'...")
    summary = download_and_mount_carinthia_on_modal.remote()
    print("Modal volume setup completed successfully!")
    print(summary)


if __name__ == "__main__":
    main()
