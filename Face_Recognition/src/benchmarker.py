"""
benchmarker.py — CPU vs GPU benchmarking for face recognition inference.
Uses the same CNN model on the same dataset images and measures:
  - Inference time per image
  - Throughput (images/sec)
  - Memory usage (RSS MB)
  - CPU utilization %
  - GPU utilization % (if available via PyTorch)
  - Speedup ratio
  - Performance across different batch sizes
"""

import os
import time
import psutil
import gc
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import (
        IMAGE_DATASET_DIR, BENCHMARK_BATCH_SIZES, BENCHMARK_REPEATS,
        VALID_EXTENSIONS
    )
except ImportError:
    from config import (
        IMAGE_DATASET_DIR, BENCHMARK_BATCH_SIZES, BENCHMARK_REPEATS,
        VALID_EXTENSIONS
    )
    pass

# ── ONNX Runtime GPU check ────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    _providers = set(ort.get_available_providers())
    GPU_AVAILABLE = bool(_providers & {
        "CUDAExecutionProvider", "DmlExecutionProvider",
        "OpenVINOExecutionProvider", "TensorrtExecutionProvider",
    })
    if "DmlExecutionProvider" in _providers:
        GPU_NAME = "Intel UHD Graphics (DirectML)"
        GPU_PROVIDER = "directml"
    elif "OpenVINOExecutionProvider" in _providers:
        GPU_NAME = "Intel GPU (OpenVINO)"
        GPU_PROVIDER = "openvino"
    elif "CUDAExecutionProvider" in _providers:
        GPU_NAME = "NVIDIA GPU (CUDA)"
        GPU_PROVIDER = "cuda"
    elif "TensorrtExecutionProvider" in _providers:
        GPU_NAME = "GPU (TensorRT)"
        GPU_PROVIDER = "tensorrt"
    else:
        GPU_NAME = "N/A"
        GPU_PROVIDER = None
except Exception:
    GPU_AVAILABLE = False
    GPU_NAME = "N/A"
    GPU_PROVIDER = None


def _collect_benchmark_images(max_images: int = 50):
    """Collect image paths from dataset(image)/ up to max_images."""
    paths = []
    if not os.path.isdir(IMAGE_DATASET_DIR):
        return paths

    for person in sorted(os.listdir(IMAGE_DATASET_DIR)):
        pdir = os.path.join(IMAGE_DATASET_DIR, person)
        if not os.path.isdir(pdir):
            continue
        for f in sorted(os.listdir(pdir)):
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(pdir, f))
                if len(paths) >= max_images:
                    return paths
    return paths


def _load_images_as_arrays(paths):
    """Load images and return as list of np.ndarray (RGB)."""
    arrays = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            arrays.append(np.array(img))
        except Exception:
            pass
    return arrays


def _measure_process():
    """Return (rss_mb, cpu_pct)."""
    proc = psutil.Process(os.getpid())
    rss_mb  = proc.memory_info().rss / (1024 ** 2)
    cpu_pct = proc.cpu_percent(interval=0.1)
    return rss_mb, cpu_pct


def _extract_face_crops(images_np: list, progress_callback=None):
    """Detect each face once so CPU and GPU time identical face crops."""
    from deepface import DeepFace

    crops = []
    for index, image in enumerate(images_np):
        if progress_callback:
            progress_callback(
                0.1 * (index + 1) / max(len(images_np), 1),
                f"Preparing face crops {index + 1}/{len(images_np)}",
            )
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for detector in ("retinaface", "opencv", "mtcnn", "ssd"):
            try:
                faces = DeepFace.extract_faces(
                    img_path=bgr_image,
                    detector_backend=detector,
                    enforce_detection=True,
                    align=True,
                    color_face="rgb",
                    normalize_face=False,
                )
                if faces:
                    face = max(
                        faces,
                        key=lambda item: item["facial_area"]["w"] * item["facial_area"]["h"],
                    )["face"]
                    if face.size and float(np.max(face)) <= 1.0:
                        face = face * 255
                    crops.append(cv2.cvtColor(np.asarray(face, dtype=np.uint8), cv2.COLOR_RGB2BGR))
                    break
            except Exception:
                continue
    return crops


def _configure_provider(provider: str):
    """Select an ONNX Runtime provider and verify the resulting FaceNet session."""
    from deepface.modules import modeling

    os.environ["DEEPFACE_ONNX_PROVIDERS"] = provider
    for task_models in getattr(modeling, "cached_models", {}).values():
        task_models.clear()
    gc.collect()

    model = modeling.build_model(task="facial_recognition", model_name="Facenet512")
    session_providers = model.model.get_providers()
    expected = "CPUExecutionProvider" if provider == "cpu" else {
        "directml": "DmlExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }[provider]
    if expected not in session_providers:
        raise RuntimeError(f"Requested {expected}, but FaceNet loaded {session_providers}")
    return session_providers


def _run_embedding_batch(images_np: list):
    """Run Facenet512 on already-detected BGR face crops."""
    from deepface import DeepFace

    start = time.perf_counter()
    DeepFace.represent(
        img_path=images_np,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    return time.perf_counter() - start


def run_benchmark(progress_callback=None):
    """Compare CPU and GPU embedding inference on identical face crops."""
    all_paths = _collect_benchmark_images(max_images=max(BENCHMARK_BATCH_SIZES))
    all_images = _load_images_as_arrays(all_paths)
    original_provider = os.environ.get("DEEPFACE_ONNX_PROVIDERS")
    results = []
    errors = {}
    crops = []

    try:
        crops = _extract_face_crops(all_images, progress_callback)
        providers_to_test = ["cpu"]
        if GPU_AVAILABLE and GPU_PROVIDER:
            providers_to_test.append(GPU_PROVIDER)

        total_steps = len(providers_to_test) * len(BENCHMARK_BATCH_SIZES) * BENCHMARK_REPEATS
        step = 0
        device_times = {}

        for provider in providers_to_test:
            device = "CPU" if provider == "cpu" else "GPU"
            try:
                active_providers = _configure_provider(provider)
                _run_embedding_batch(crops[:1])
                for batch_size in BENCHMARK_BATCH_SIZES:
                    batch = crops[:batch_size]
                    if not batch:
                        continue

                    elapsed_times = []
                    cpu_loads = []
                    for rep in range(BENCHMARK_REPEATS):
                        step += 1
                        if progress_callback:
                            progress_callback(
                                0.1 + 0.9 * step / max(total_steps, 1),
                                f"{device} ({active_providers[0]}) | batch={len(batch)} | "
                                f"rep {rep + 1}/{BENCHMARK_REPEATS}",
                            )
                        gc.collect()
                        elapsed_times.append(_run_embedding_batch(batch))
                        _, cpu_pct = _measure_process()
                        cpu_loads.append(cpu_pct)

                    avg_time = float(np.mean(elapsed_times))
                    results.append({
                        "batch_size": batch_size,
                        "actual_batch": len(batch),
                        "device": device,
                        "provider": active_providers[0],
                        "avg_time_per_image_ms": round(avg_time / len(batch) * 1000, 2),
                        "throughput_imgs_sec": round(len(batch) / avg_time, 2) if avg_time > 0 else 0,
                        "total_time_sec": round(avg_time, 4),
                        "memory_mb": round(
                            psutil.Process(os.getpid()).memory_info().rss / 1024**2, 1
                        ),
                        "cpu_pct": round(float(np.mean(cpu_loads)), 1),
                        "gpu_pct": None,
                    })
                    device_times.setdefault(device, {})[batch_size] = avg_time
            except Exception as exc:
                errors[device] = str(exc)

        cpu_times = device_times.get("CPU", {})
        gpu_times = device_times.get("GPU", {})
        speedup_by_batch = {
            size: round(cpu_time / gpu_times[size], 2)
            for size, cpu_time in cpu_times.items()
            if size in gpu_times and gpu_times[size] > 0
        }
        return {
            "images_available": len(all_images),
            "faces_available": len(crops),
            "gpu_available": GPU_AVAILABLE,
            "gpu_name": GPU_NAME,
            "results": results,
            "speedup_by_batch": speedup_by_batch,
            "errors": errors,
        }
    finally:
        from deepface.modules import modeling
        for task_models in getattr(modeling, "cached_models", {}).values():
            task_models.clear()
        if GPU_AVAILABLE and GPU_PROVIDER:
            os.environ["DEEPFACE_ONNX_PROVIDERS"] = f"{GPU_PROVIDER},cpu"
        elif original_provider is None:
            os.environ.pop("DEEPFACE_ONNX_PROVIDERS", None)
        else:
            os.environ["DEEPFACE_ONNX_PROVIDERS"] = original_provider
        gc.collect()


if __name__ == "__main__":
    def cb(p, msg):
        bar = "█" * int(p * 30)
        print(f"\r[{bar:<30}] {p*100:.0f}% — {msg}", end="", flush=True)

    print("Starting benchmark...")
    report = run_benchmark(progress_callback=cb)
    print("\n")
    print(f"Images available : {report['images_available']}")
    print(f"GPU available    : {report['gpu_available']} ({report['gpu_name']})")
    print(f"\n{'Device':<6} {'Batch':>5} {'ms/img':>8} {'img/s':>8} {'Mem MB':>8} {'CPU%':>6}")
    print("-" * 50)
    for r in report["results"]:
        print(f"{r['device']:<6} {r['batch_size']:>5} "
              f"{r['avg_time_per_image_ms']:>8.2f} "
              f"{r['throughput_imgs_sec']:>8.2f} "
              f"{r['memory_mb']:>8.1f} "
              f"{str(r['cpu_pct']):>6}")
    print("\nSpeedup (GPU/CPU):", report["speedup_by_batch"])
