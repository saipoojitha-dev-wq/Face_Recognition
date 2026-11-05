"""
config.py — Central configuration for the Face Recognition System
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DATASET_DIR   = os.path.join(BASE_DIR, "dataset(image)")
CSV_DATASET_PATH    = os.path.join(BASE_DIR, "Dataset.csv")
PROCESSED_DIR       = os.path.join(BASE_DIR, "processed")
EMBEDDINGS_DB_DIR   = os.path.join(BASE_DIR, "embeddings_db")
EMBEDDINGS_PKL      = os.path.join(EMBEDDINGS_DB_DIR, "embeddings.pkl")
SPLIT_INFO_PKL      = os.path.join(EMBEDDINGS_DB_DIR, "split_info.pkl")

# ── Preprocessing ──────────────────────────────────────────────────────────────
FACE_SIZE           = (160, 160)       # target crop size (w, h)
# DeepFace backend: "opencv" (fast, no dlib), "retinaface" (accurate), "mtcnn" (balanced)
DEEPFACE_DETECTOR   = "opencv"
DEEPFACE_MODEL      = "Facenet512"     # 512-dim embeddings; accurate & pre-compiled ONNX

# ── Recognition ───────────────────────────────────────────────────────────────
# L2-normalized cosine distance threshold (Facenet512: good range 0.30–0.50)
RECOGNITION_THRESHOLD = 0.85
REGISTRATION_SPLIT    = 0.70          # fraction of images used for registration

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_BATCH_SIZES = [1, 4, 8, 16]
BENCHMARK_REPEATS     = 3             # repeat each batch size N times and average

# ── Supported image extensions ────────────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Ensure output directories exist ───────────────────────────────────────────
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DB_DIR, exist_ok=True)
