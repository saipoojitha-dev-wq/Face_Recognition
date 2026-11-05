# Face Recognition System

A complete face recognition system built on two datasets:
1. **Custom personal dataset** (`dataset(image)/`) — 4 people, 18 images
2. **Online celebrity CSV dataset** (`Dataset.csv`) — 2563 entries, ~35 celebrities

## Project Structure

```
Face_Recognition/
├── Dataset.csv              ← Online celebrity CSV dataset
├── dataset(image)/          ← Custom personal image dataset (4 persons)
│   ├── aradhya/             ← 2 images
│   ├── keerthi/             ← 5 images
│   ├── saipoojitha/         ← 6 images
│   └── varalakshmi/         ← 5 images
├── processed/               ← [Auto-created] Preprocessed face crops
├── embeddings_db/           ← [Auto-created] Saved embedding database
├── src/
│   ├── config.py            ← All config constants and paths
│   ├── dataset_inspector.py ← Dataset scanning and validation
│   ├── preprocessor.py      ← Face detection, crop, resize, normalize
│   ├── embedder.py          ← 128-dim face embedding generation
│   ├── registrar.py         ← Train/test split + embedding database
│   ├── recognizer.py        ← Identify faces against registered DB
│   └── benchmarker.py       ← CPU vs GPU performance benchmarking
├── app.py                   ← Streamlit web dashboard (5 tabs)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

On Windows, `requirements.txt` installs ONNX Runtime DirectML for GPU inference
on supported Intel, NVIDIA, and AMD graphics hardware. DeepFace falls back to
CPU when DirectML is unavailable. On other platforms, the requirements install
the standard ONNX Runtime CPU package.

## Running the App

```bash
streamlit run app.py
```

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Dataset Overview | Total people, images, valid/invalid, faces detected, CSV breakdown |
| 🔍 Inspect & Preprocess | Per-person image inspection + face preprocessing |
| 📝 Register Faces | Build embedding database, view train/test split, evaluate accuracy |
| 🎯 Recognize Face | Upload any image → get matched identity or "Unknown" |
| ⚡ Benchmark | Paired CPU/GPU latency, throughput, memory, and measured speedup |
