"""
embedder.py — Generate face embeddings using DeepFace (Facenet512 model, ONNX backend).
No compilation required — works on Python 3.14 without Visual C++.
"""

import os
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
    if "DmlExecutionProvider" in ort.get_available_providers():
        os.environ.setdefault("DEEPFACE_ONNX_PROVIDERS", "directml,cpu")
except Exception:
    pass

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import VALID_EXTENSIONS
except ImportError:
    from config import VALID_EXTENSIONS

# DeepFace model — Facenet512 gives 512-dim embeddings; very accurate, pre-compiled ONNX
DEEPFACE_MODEL    = "Facenet512"
DEEPFACE_DETECTOR = "opencv"   # fast built-in detector; no extra install


def _represent_with_detector_fallback(image_path: str):
    from deepface import DeepFace

    errors = []
    for detector in (DEEPFACE_DETECTOR, "retinaface", "mtcnn", "ssd"):
        try:
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name=DEEPFACE_MODEL,
                detector_backend=detector,
                enforce_detection=True,
                align=True,
            )
            if embeddings:
                return embeddings
        except Exception as exc:
            errors.append(f"{detector}: {exc}")

    raise ValueError("No face could be embedded. " + " | ".join(errors))


def embed_image(image_path: str = None, image_np: np.ndarray = None):
    """
    Generate a face embedding from an image file or numpy array.

    Parameters
    ----------
    image_path : str (optional) — path to image file
    image_np   : np.ndarray (optional) — RGB image array

    Returns
    -------
    dict:
        success   : bool
        embedding : np.ndarray (512,) or None
        reason    : str error message if failed
    """
    result = {"success": False, "embedding": None, "reason": None}

    # ── Resolve image source ──────────────────────────────────────────────────
    if image_path is None and image_np is None:
        result["reason"] = "No image provided"
        return result

    # Save np array to temp file if no path given
    _tmp_path = None
    if image_path is None:
        import tempfile, cv2
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        _tmp_path = tmp.name
        tmp.close()
        cv2.imwrite(_tmp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        image_path = _tmp_path

    # ── Run DeepFace embedding ────────────────────────────────────────────────
    try:
        embeddings = _represent_with_detector_fallback(image_path)
        if not embeddings:
            result["reason"] = "No face detected"
            return result

        # Pick the largest face
        if len(embeddings) > 1:
            best = max(embeddings, key=lambda x: x.get("face_confidence", 0))
        else:
            best = embeddings[0]

        emb = np.array(best["embedding"], dtype=np.float32)
        # L2-normalise so cosine ~ euclidean distance
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        result["success"]   = True
        result["embedding"] = emb
        return result

    except Exception as e:
        result["reason"] = f"Embedding error: {e}"
        return result
    finally:
        if _tmp_path and os.path.exists(_tmp_path):
            os.remove(_tmp_path)


def embed_folder(folder_path: str, progress_callback=None):
    """
    Generate embeddings for all images in a flat folder.

    Returns
    -------
    list of dicts: [{filename, success, embedding, reason}]
    """
    results = []
    files = [
        f for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    ]

    for idx, fname in enumerate(files):
        fpath = os.path.join(folder_path, fname)
        if progress_callback:
            progress_callback((idx+1)/max(len(files),1), f"Embedding {fname}")

        res = embed_image(image_path=fpath)
        results.append({
            "filename":  fname,
            "success":   res["success"],
            "embedding": res["embedding"],
            "reason":    res["reason"],
        })

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        res = embed_image(image_path=path)
        if res["success"]:
            emb = res["embedding"]
            print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
            print(f"First 8 dims: {emb[:8]}")
        else:
            print(f"Failed: {res['reason']}")
