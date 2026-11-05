"""
preprocessor.py — Face detection, cropping, alignment & resizing using DeepFace + OpenCV.
Saves processed faces to processed/<person>/<filename> without touching originals.
"""

import os
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

warnings.filterwarnings("ignore")

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import (
        IMAGE_DATASET_DIR, PROCESSED_DIR, VALID_EXTENSIONS, FACE_SIZE
    )
except ImportError:
    from config import (
        IMAGE_DATASET_DIR, PROCESSED_DIR, VALID_EXTENSIONS, FACE_SIZE
    )

# Use DeepFace's built-in OpenCV detector (no dlib needed)
DEEPFACE_DETECTOR = "opencv"
DEEPFACE_MODEL    = "Facenet512"


def preprocess_image(image_path: str, save_path: str = None):
    """
    Load image → detect face → crop & align → resize → normalize.

    Parameters
    ----------
    image_path : str  — path to source image
    save_path  : str  — if given, saves cropped face PNG here

    Returns
    -------
    dict:
        success        : bool
        face_array     : np.ndarray (H, W, 3) normalized float32  or None
        face_pil       : PIL.Image (RGB, 160×160)                  or None
        location       : (top, right, bottom, left) tuple          or None
        reason         : str error message if failed
    """
    result = {
        "success":    False,
        "face_array": None,
        "face_pil":   None,
        "location":   None,
        "reason":     None,
    }

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_np  = np.array(pil_img)
    except Exception as e:
        result["reason"] = f"Cannot load image: {e}"
        return result

    # ── Detect & extract face ─────────────────────────────────────────────────
    face_crop = None

    # Strategy 1: DeepFace (primary — no dlib, uses ONNX)
    try:
        from deepface import DeepFace
        face_objs = DeepFace.extract_faces(
            img_path          = image_path,
            detector_backend  = DEEPFACE_DETECTOR,
            enforce_detection = True,
            align             = True,
        )
        if face_objs:
            best = max(face_objs, key=lambda f: f.get("confidence", 0))
            face_arr = best["face"]
            if face_arr.dtype != np.uint8:
                face_arr = (face_arr * 255).clip(0, 255).astype(np.uint8)
            face_crop = face_arr
            fa = best.get("facial_area", {})
            result["location"] = (fa.get("y", 0), fa.get("x", 0) + fa.get("w", 0),
                                   fa.get("y", 0) + fa.get("h", 0), fa.get("x", 0))
    except Exception:
        pass

    # Strategy 2: OpenCV Haar cascade fallback (safe load with existence check)
    if face_crop is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier()
        loaded = cascade.load(cascade_path)
        if loaded and not cascade.empty():
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                mx, my = int(w * 0.10), int(h * 0.10)
                x1 = max(0, x - mx);  y1 = max(0, y - my)
                x2 = min(img_np.shape[1], x + w + mx)
                y2 = min(img_np.shape[0], y + h + my)
                face_crop = img_np[y1:y2, x1:x2]
                result["location"] = (y1, x2, y2, x1)

    # Strategy 3: Use the full image (no face crop) — avoids hard failure
    if face_crop is None or face_crop.size == 0:
        # Soft fallback: centre-crop the full image
        h, w = img_np.shape[:2]
        margin = int(min(h, w) * 0.05)
        face_crop = img_np[margin:h-margin, margin:w-margin]
        result["location"] = None
        result["reason"]   = "No face detected — using full image region"

    # ── Resize ────────────────────────────────────────────────────────────────
    face_resized = cv2.resize(face_crop, FACE_SIZE, interpolation=cv2.INTER_AREA)

    # ── Normalize ─────────────────────────────────────────────────────────────
    face_norm = face_resized.astype(np.float32) / 255.0

    # ── PIL image ─────────────────────────────────────────────────────────────
    face_pil = Image.fromarray(face_resized)

    result["success"]    = True
    result["face_array"] = face_norm
    result["face_pil"]   = face_pil

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        face_pil.save(save_path)

    return result


def preprocess_dataset(progress_callback=None):
    """
    Preprocess every image in dataset(image)/ and save to processed/.

    Returns
    -------
    dict:
        processed : {person: [{"source", "dest", "success", "reason"}]}
        summary   : {total, succeeded, failed}
    """
    report = {"processed": {}, "summary": {"total": 0, "succeeded": 0, "failed": 0}}

    if not os.path.isdir(IMAGE_DATASET_DIR):
        return report

    persons = sorted([
        d for d in os.listdir(IMAGE_DATASET_DIR)
        if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, d))
    ])

    all_files = []
    for person in persons:
        pdir = os.path.join(IMAGE_DATASET_DIR, person)
        for f in os.listdir(pdir):
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                all_files.append((person, f))

    total = len(all_files)

    for idx, (person, fname) in enumerate(all_files):
        src = os.path.join(IMAGE_DATASET_DIR, person, fname)
        dest_name = Path(fname).stem + ".png"
        dest = os.path.join(PROCESSED_DIR, person, dest_name)

        if progress_callback:
            progress_callback(
                (idx + 1) / max(total, 1),
                f"Processing {person}/{fname}"
            )

        res = preprocess_image(src, save_path=dest)

        entry = {
            "source":  src,
            "dest":    dest if res["success"] else None,
            "success": res["success"],
            "reason":  res["reason"],
        }

        report["processed"].setdefault(person, []).append(entry)
        report["summary"]["total"] += 1
        if res["success"]:
            report["summary"]["succeeded"] += 1
        else:
            report["summary"]["failed"] += 1

    return report


if __name__ == "__main__":
    def cb(p, msg):
        print(f"[{p*100:5.1f}%] {msg}")

    result = preprocess_dataset(progress_callback=cb)
    s = result["summary"]
    print(f"\nDone — Total:{s['total']}  OK:{s['succeeded']}  Failed:{s['failed']}")
    for person, entries in result["processed"].items():
        ok  = sum(1 for e in entries if e["success"])
        bad = sum(1 for e in entries if not e["success"])
        print(f"  {person:20s} → {ok} OK, {bad} failed")
        for e in entries:
            if not e["success"]:
                print(f"      ✗ {e['source']} → {e['reason']}")
