"""
recognizer.py — Compare a query image against the registered embedding database.
Returns best-match identity or "Unknown".
"""

import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.config import RECOGNITION_THRESHOLD
    from src.registrar import load_registration_database
    from src.embedder import embed_image
except ImportError:
    from config import RECOGNITION_THRESHOLD
    from registrar import load_registration_database
    from embedder import embed_image


def recognize_face(image_path: str = None, image_np: np.ndarray = None,
                   registration_db: dict = None, threshold: float = None):
    """
    Identify who is in the given image.

    Parameters
    ----------
    image_path      : str           — path to query image
    image_np        : np.ndarray    — RGB image array (alternative to path)
    registration_db : dict          — {name: [emb, ...]}  (loaded if None)
    threshold       : float         — override default distance threshold

    Returns
    -------
    dict:
        success      : bool
        identity     : str  — predicted name or "Unknown"
        confidence   : float — 0..1  (1 = perfect match)
        distance     : float — Euclidean distance to nearest neighbour
        all_distances: {name: min_distance}
        reason       : str error message if failed
    """
    result = {
        "success":        False,
        "identity":       "Unknown",
        "confidence":     0.0,
        "distance":       float("inf"),
        "all_distances":  {},
        "face_location":  None,   # (top, right, bottom, left) for bounding box
        "reason":         None,
    }

    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    # ── Load DB if not provided ───────────────────────────────────────────────
    if registration_db is None:
        registration_db, _ = load_registration_database()
        if registration_db is None:
            result["reason"] = "No registration database found. Please register faces first."
            return result

    if not registration_db:
        result["reason"] = "Registration database is empty."
        return result

    # ── Embed query image & extract face location ─────────────────────────────
    emb_result = embed_image(image_path=image_path, image_np=image_np)
    if not emb_result["success"]:
        result["reason"] = emb_result["reason"]
        return result

    query_emb = emb_result["embedding"]

    # Try to get face location for bounding box drawing
    try:
        _src = image_path
        if _src is None and image_np is not None:
            import tempfile, cv2 as _cv2
            _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            _src = _tmp.name; _tmp.close()
            _cv2.imwrite(_src, _cv2.cvtColor(image_np, _cv2.COLOR_RGB2BGR))
        from deepface import DeepFace as _DF
        faces = _DF.extract_faces(img_path=_src, detector_backend="opencv",
                                   enforce_detection=False, align=False)
        if faces:
            fa = faces[0].get("facial_area", {})
            x, y, w, h = fa.get("x",0), fa.get("y",0), fa.get("w",0), fa.get("h",0)
            result["face_location"] = (y, x+w, y+h, x)   # top, right, bottom, left
    except Exception:
        pass

    # ── Compare against all registered embeddings ─────────────────────────────
    all_distances = {}
    for name, emb_list in registration_db.items():
        dists = [np.linalg.norm(query_emb - e) for e in emb_list]
        all_distances[name] = round(float(min(dists)), 4)

    result["all_distances"] = all_distances

    # ── Find best match ───────────────────────────────────────────────────────
    best_name = min(all_distances, key=all_distances.get)
    best_dist = all_distances[best_name]

    result["distance"] = best_dist
    result["success"]  = True

    if best_dist <= threshold:
        result["identity"]   = best_name
        # Map distance to confidence: 0 dist → 100%, threshold dist → ~0%
        result["confidence"] = round(float(max(0.0, 1.0 - (best_dist / threshold))), 4)
    else:
        result["identity"]   = "Unknown"
        result["confidence"] = 0.0

    return result


def evaluate_on_test_set(registration_db: dict = None, split_info: dict = None,
                          progress_callback=None):
    """
    Run recognition on the test-set images and compute accuracy.

    Returns
    -------
    dict:
        accuracy    : float
        correct     : int
        total       : int
        per_person  : {name: {correct, total, accuracy}}
        predictions : list of {image, true, predicted, correct, distance}
    """
    if registration_db is None or split_info is None:
        from src.registrar import load_registration_database
        registration_db, split_info = load_registration_database()

    if registration_db is None:
        return {"error": "No database loaded"}

    predictions = []
    per_person  = {}
    total = correct = 0

    all_test = [
        (person, fpath)
        for person, info in split_info.items()
        for fpath in info.get("test", [])
    ]

    for idx, (true_name, fpath) in enumerate(all_test):
        if progress_callback:
            progress_callback((idx+1)/max(len(all_test),1),
                              f"Evaluating {os.path.basename(fpath)}")

        res = recognize_face(image_path=fpath, registration_db=registration_db)
        predicted = res["identity"]
        is_correct = (predicted == true_name)

        predictions.append({
            "image":     fpath,
            "true":      true_name,
            "predicted": predicted,
            "correct":   is_correct,
            "distance":  res["distance"],
            "confidence": res["confidence"],
        })

        per_person.setdefault(true_name, {"correct": 0, "total": 0})
        per_person[true_name]["total"]   += 1
        per_person[true_name]["correct"] += int(is_correct)
        total   += 1
        correct += int(is_correct)

    for name in per_person:
        t = per_person[name]["total"]
        c = per_person[name]["correct"]
        per_person[name]["accuracy"] = round(c / t, 4) if t > 0 else 0.0

    return {
        "accuracy":    round(correct / total, 4) if total > 0 else 0.0,
        "correct":     correct,
        "total":       total,
        "per_person":  per_person,
        "predictions": predictions,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Recognizing: {path}")
        res = recognize_face(image_path=path)
        print(f"  Identity  : {res['identity']}")
        print(f"  Confidence: {res['confidence']*100:.1f}%")
        print(f"  Distance  : {res['distance']:.4f}")
        if res["all_distances"]:
            print("  All distances:")
            for name, d in sorted(res["all_distances"].items(), key=lambda x: x[1]):
                print(f"    {name:25s} → {d:.4f}")
