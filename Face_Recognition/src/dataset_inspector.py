"""
dataset_inspector.py — Inspect both datasets and return a structured summary.
Handles:
  1. dataset(image)/ — folder-per-person image dataset
  2. Dataset.csv     — online celebrity CSV dataset (filename → label)
"""

import os
import io
import csv
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import IMAGE_DATASET_DIR, CSV_DATASET_PATH, VALID_EXTENSIONS
    from src.embedder import embed_image
except ImportError:
    from config import IMAGE_DATASET_DIR, CSV_DATASET_PATH, VALID_EXTENSIONS
    from embedder import embed_image

# OpenCV Haar cascade — load safely with existence check
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier()
_CASCADE_OK = (
    os.path.isfile(_CASCADE_PATH)
    and _face_cascade.load(_CASCADE_PATH)
    and not _face_cascade.empty()
)


def _detect_face_opencv(img_np):
    """Return True if at least one face is detected.
    Uses Haar cascade when available, otherwise tries DeepFace."""
    # Try Haar cascade first (fast)
    if _CASCADE_OK:
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = _face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            if len(faces) > 0:
                return True
        except Exception:
            pass

    # Use the same detector/model fallback as registration and recognition.
    return embed_image(image_np=img_np)["success"]


# ──────────────────────────────────────────────────────────────────────────────
# Image Dataset Inspector
# ──────────────────────────────────────────────────────────────────────────────

def inspect_image_dataset(progress_callback=None):
    """
    Walk dataset(image)/ folder and return per-person + global stats.

    Returns
    -------
    dict with keys:
        people          : list of person names
        per_person      : {name: {images, valid, invalid, faces_detected, files}}
        total_images    : int
        total_valid     : int
        total_invalid   : int
        total_faces     : int
        invalid_files   : list of (path, reason)
    """
    result = {
        "people": [],
        "per_person": {},
        "total_images": 0,
        "total_valid": 0,
        "total_invalid": 0,
        "total_faces": 0,
        "invalid_files": [],
    }

    if not os.path.isdir(IMAGE_DATASET_DIR):
        return result

    persons = sorted([
        d for d in os.listdir(IMAGE_DATASET_DIR)
        if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, d))
    ])

    result["people"] = persons
    total_steps = sum(
        len(os.listdir(os.path.join(IMAGE_DATASET_DIR, p))) for p in persons
    )
    step = 0

    for person in persons:
        person_dir = os.path.join(IMAGE_DATASET_DIR, person)
        files = [
            f for f in os.listdir(person_dir)
            if Path(f).suffix.lower() in VALID_EXTENSIONS
        ]

        person_stats = {
            "images": len(files),
            "valid": 0,
            "invalid": 0,
            "faces_detected": 0,
            "files": [],
        }

        for fname in files:
            fpath = os.path.join(person_dir, fname)
            step += 1

            if progress_callback:
                progress_callback(step / max(total_steps, 1),
                                  f"Inspecting {person}/{fname}")

            file_info = {
                "name": fname,
                "path": fpath,
                "valid": False,
                "face_detected": False,
                "reason": None,
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
            }

            # 1. Try loading image
            try:
                img = Image.open(fpath)
                img.verify()          # checks file integrity
                img = Image.open(fpath).convert("RGB")
                img_np = np.array(img)
                file_info["valid"] = True
                person_stats["valid"] += 1
            except Exception as e:
                file_info["reason"] = f"Corrupted: {e}"
                person_stats["invalid"] += 1
                result["invalid_files"].append((fpath, str(e)))
                person_stats["files"].append(file_info)
                continue

            # 2. Try face detection (OpenCV Haar cascade — no dlib required)
            try:
                face_found = _detect_face_opencv(img_np)
                if face_found:
                    file_info["face_detected"] = True
                    person_stats["faces_detected"] += 1
                    result["total_faces"] += 1
                else:
                    file_info["reason"] = "No face detected"
            except Exception as e:
                file_info["reason"] = f"Detection error: {e}"

            person_stats["files"].append(file_info)

        result["per_person"][person] = person_stats
        result["total_images"] += person_stats["images"]
        result["total_valid"]  += person_stats["valid"]
        result["total_invalid"] += person_stats["invalid"]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CSV Dataset Inspector
# ──────────────────────────────────────────────────────────────────────────────

def inspect_csv_dataset():
    """
    Parse Dataset.csv and return per-label statistics.

    Returns
    -------
    dict with keys:
        total_entries   : int
        unique_labels   : int
        per_label       : {label: count}
        sample_entries  : list of (filename, label) — first 10
        columns         : list of column names
    """
    result = {
        "total_entries": 0,
        "unique_labels": 0,
        "per_label": {},
        "sample_entries": [],
        "columns": [],
    }

    if not os.path.isfile(CSV_DATASET_PATH):
        return result

    with open(CSV_DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result["columns"] = reader.fieldnames or []

        for i, row in enumerate(reader):
            label = row.get("label", "").strip()
            filename = row.get("id", "").strip()

            if not label:
                continue

            result["total_entries"] += 1
            result["per_label"][label] = result["per_label"].get(label, 0) + 1

            if i < 10:
                result["sample_entries"].append((filename, label))

    result["unique_labels"] = len(result["per_label"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Combined Summary
# ──────────────────────────────────────────────────────────────────────────────

def full_inspection_report(progress_callback=None):
    """Run both inspectors and return combined results."""
    img_summary = inspect_image_dataset(progress_callback=progress_callback)
    csv_summary = inspect_csv_dataset()
    return {"image_dataset": img_summary, "csv_dataset": csv_summary}


if __name__ == "__main__":
    report = full_inspection_report()
    img = report["image_dataset"]
    csv = report["csv_dataset"]

    print("=" * 60)
    print("IMAGE DATASET SUMMARY")
    print("=" * 60)
    print(f"  People        : {len(img['people'])} → {img['people']}")
    print(f"  Total images  : {img['total_images']}")
    print(f"  Valid images  : {img['total_valid']}")
    print(f"  Invalid       : {img['total_invalid']}")
    print(f"  Faces detected: {img['total_faces']}")
    for person, stats in img["per_person"].items():
        print(f"  {person:20s} → {stats['images']} images, "
              f"{stats['faces_detected']} faces detected")

    print("\n" + "=" * 60)
    print("CSV DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total entries  : {csv['total_entries']}")
    print(f"  Unique labels  : {csv['unique_labels']}")
    for label, cnt in sorted(csv["per_label"].items()):
        print(f"  {label:30s} → {cnt} images")
