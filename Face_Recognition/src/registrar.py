"""
registrar.py — Build embedding database from dataset(image)/.
Performs identity-aware train/test split, embeds registration images,
and saves the embedding database to embeddings_db/embeddings.pkl.
"""

import os
import math
import pickle
import random
import numpy as np
from pathlib import Path

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import (
        IMAGE_DATASET_DIR, EMBEDDINGS_PKL, SPLIT_INFO_PKL,
        REGISTRATION_SPLIT, VALID_EXTENSIONS
    )
    from src.embedder import embed_image
except ImportError:
    from config import (
        IMAGE_DATASET_DIR, EMBEDDINGS_PKL, SPLIT_INFO_PKL,
        REGISTRATION_SPLIT, VALID_EXTENSIONS
    )
    from embedder import embed_image


def build_registration_database(progress_callback=None, seed=42):
    """
    1. Enumerate all images in dataset(image)/
    2. Perform identity-aware train/test split (REGISTRATION_SPLIT fraction → register)
    3. Embed registration images
    4. Save {person: [emb1, emb2, ...]} to EMBEDDINGS_PKL
    5. Save split info to SPLIT_INFO_PKL

    Returns
    -------
    dict:
        registration_db  : {person: [np.ndarray]}    — registered embeddings
        split_info       : {person: {register:[], test:[]}}
        summary          : {total, registered, failed, people}
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.isdir(IMAGE_DATASET_DIR):
        return {"error": f"Dataset directory not found: {IMAGE_DATASET_DIR}"}

    persons = sorted([
        d for d in os.listdir(IMAGE_DATASET_DIR)
        if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, d))
    ])

    split_info      = {}
    registration_db = {}
    summary = {"total": 0, "registered": 0, "failed": 0, "people": len(persons)}

    # ── Collect all files per person ──────────────────────────────────────────
    for person in persons:
        pdir  = os.path.join(IMAGE_DATASET_DIR, person)
        files = sorted([
            f for f in os.listdir(pdir)
            if Path(f).suffix.lower() in VALID_EXTENSIONS
        ])

        if not files:
            split_info[person] = {"register": [], "test": []}
            continue

        random.shuffle(files)
        n_register = max(1, math.ceil(len(files) * REGISTRATION_SPLIT))
        register_files = files[:n_register]
        test_files     = files[n_register:]

        split_info[person] = {
            "register": [os.path.join(pdir, f) for f in register_files],
            "test":     [os.path.join(pdir, f) for f in test_files],
        }

    # ── Embed registration images ─────────────────────────────────────────────
    all_reg_files = [
        (person, fpath)
        for person, info in split_info.items()
        for fpath in info["register"]
    ]
    total = len(all_reg_files)

    for idx, (person, fpath) in enumerate(all_reg_files):
        if progress_callback:
            progress_callback(
                (idx + 1) / max(total, 1),
                f"Registering {person} ← {os.path.basename(fpath)}"
            )

        res = embed_image(image_path=fpath)
        summary["total"] += 1

        if res["success"]:
            registration_db.setdefault(person, []).append(res["embedding"])
            summary["registered"] += 1
        else:
            summary["failed"] += 1

    # ── Save to disk ──────────────────────────────────────────────────────────
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(registration_db, f)

    with open(SPLIT_INFO_PKL, "wb") as f:
        pickle.dump(split_info, f)

    return {
        "registration_db": registration_db,
        "split_info":      split_info,
        "summary":         summary,
    }


def load_registration_database():
    """Load previously saved embedding database."""
    if not os.path.isfile(EMBEDDINGS_PKL):
        return None, None

    with open(EMBEDDINGS_PKL, "rb") as f:
        db = pickle.load(f)

    split_info = None
    if os.path.isfile(SPLIT_INFO_PKL):
        with open(SPLIT_INFO_PKL, "rb") as f:
            split_info = pickle.load(f)

    return db, split_info


def registration_status():
    """Return registration status summary."""
    db, split_info = load_registration_database()
    dataset_people = []
    if os.path.isdir(IMAGE_DATASET_DIR):
        dataset_people = sorted([
            person for person in os.listdir(IMAGE_DATASET_DIR)
            if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, person))
            and any(
                Path(filename).suffix.lower() in VALID_EXTENSIONS
                for filename in os.listdir(os.path.join(IMAGE_DATASET_DIR, person))
            )
        ])

    if db is None:
        return {
            "registered": False,
            "dataset_ready": bool(dataset_people),
            "people": dataset_people,
            "total_embeddings": 0,
            "per_person": {},
        }

    people = list(db.keys())
    total  = sum(len(v) for v in db.values())
    return {
        "registered": True,
        "dataset_ready": bool(dataset_people),
        "people":     people,
        "total_embeddings": total,
        "per_person": {p: len(v) for p, v in db.items()},
    }


if __name__ == "__main__":
    def cb(p, msg):
        print(f"[{p*100:5.1f}%] {msg}")

    result = build_registration_database(progress_callback=cb)
    if "error" in result:
        print("Error:", result["error"])
    else:
        s = result["summary"]
        print(f"\nRegistration complete:")
        print(f"  People     : {s['people']}")
        print(f"  Total imgs : {s['total']}")
        print(f"  Registered : {s['registered']}")
        print(f"  Failed     : {s['failed']}")
        print(f"\nEmbedding DB saved to: {EMBEDDINGS_PKL}")
        for person, embs in result["registration_db"].items():
            print(f"  {person:20s} → {len(embs)} embeddings registered")
