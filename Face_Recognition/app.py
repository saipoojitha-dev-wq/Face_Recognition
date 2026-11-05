"""
app.py — Face Recognition System Dashboard (Streamlit)
6-Tab Application:
  1. 📊 Dataset Overview
  2. 🔍 Inspect & Preprocess
  3. 📝 Register Faces
  4. 📸 Live Webcam Recognition  ← NEW (shows face name + CPU/GPU perf)
  5. 🎯 Recognize Face (upload)
    6. ⚡ Inference Benchmark
"""

import os
import sys
import io
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    IMAGE_DATASET_DIR, CSV_DATASET_PATH, PROCESSED_DIR,
    EMBEDDINGS_PKL, SPLIT_INFO_PKL, RECOGNITION_THRESHOLD
)
from src.dataset_inspector import inspect_image_dataset, inspect_csv_dataset
from src.preprocessor      import preprocess_dataset
from src.registrar         import (
    build_registration_database, load_registration_database, registration_status
)
from src.recognizer        import recognize_face, evaluate_on_test_set
from src.benchmarker       import run_benchmark, GPU_AVAILABLE, GPU_NAME

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    color: white;
  }
  section[data-testid="stSidebar"] * { color: white !important; }
  .main { background: #0d1117; }

  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
  }
  div[data-testid="metric-container"]:hover { transform: translateY(-2px); }
  div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(31,111,235,0.3);
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
  }
  .section-header {
    background: linear-gradient(135deg, #1f6feb22, #388bfd11);
    border-left: 4px solid #1f6feb;
    border-radius: 0 8px 8px 0;
    padding: 12px 20px;
    margin: 24px 0 16px 0;
    color: #58a6ff;
    font-weight: 700;
    font-size: 1.1rem;
  }

  /* Webcam result cards */
  .result-card-match {
    background: linear-gradient(135deg, #0d2818, #0f3822);
    border: 2px solid #3fb950;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
  }
  .result-card-unknown {
    background: linear-gradient(135deg, #2d1515, #3d1a1a);
    border: 2px solid #f85149;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
  }
  .name-display {
    font-size: 2.4rem;
    font-weight: 800;
    color: #3fb950;
    text-shadow: 0 0 20px rgba(63,185,80,0.5);
    margin: 10px 0;
  }
  .name-display-unknown {
    font-size: 2.4rem;
    font-weight: 800;
    color: #f85149;
    text-shadow: 0 0 20px rgba(248,81,73,0.5);
    margin: 10px 0;
  }
  .perf-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }
  .perf-label { color: #8b949e; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
  .perf-value-cpu { color: #58a6ff; font-size: 1.6rem; font-weight: 700; }
  .perf-value-gpu { color: #3fb950; font-size: 1.6rem; font-weight: 700; }
  .perf-speedup   { color: #d29922; font-size: 1.6rem; font-weight: 700; }

  div.stProgress > div > div > div { background: linear-gradient(90deg, #1f6feb, #388bfd); }
  button[data-baseweb="tab"] { color: #8b949e !important; font-weight: 600; }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #1f6feb !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper — annotate image with bounding box + name
# ══════════════════════════════════════════════════════════════════════════════
def annotate_image(img_np: np.ndarray, identity: str, confidence: float,
                   face_location=None) -> np.ndarray:
    """Draw bounding box and name label on the image."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    color   = (63, 185, 80) if identity != "Unknown" else (73, 81, 248)  # BGR
    label   = f"{identity}  {confidence*100:.0f}%" if identity != "Unknown" else "Unknown"

    # If face location available, draw box; otherwise draw full border
    if face_location:
        top, right, bottom, left = face_location
        # Clamp
        top    = max(0, int(top))
        right  = min(w, int(right))
        bottom = min(h, int(bottom))
        left   = max(0, int(left))

        # Draw rounded-ish thick rectangle
        cv2.rectangle(img_bgr, (left, top), (right, bottom), color, 3)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_bg_top = max(top - th - 14, 0)
        cv2.rectangle(img_bgr, (left, label_bg_top), (left + tw + 10, top), color, -1)
        cv2.putText(img_bgr, label, (left + 5, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # Draw border around full image
        border = 6
        cv2.rectangle(img_bgr, (border, border), (w-border, h-border), color, border)
        # Name at bottom
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img_bgr, (0, h - th - 20), (tw + 20, h), color, -1)
        cv2.putText(img_bgr, label, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# Helper — timed recognition (CPU + GPU)
# ══════════════════════════════════════════════════════════════════════════════
def recognize_with_timing(img_np: np.ndarray, registration_db: dict, threshold: float):
    """Run recognition once and report the active inference backend and latency."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        started = time.perf_counter()
        result = recognize_face(
            image_path=tmp_path,
            registration_db=registration_db,
            threshold=threshold,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result, elapsed_ms, GPU_NAME if GPU_AVAILABLE else "CPU"
    finally:
        os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 Face Recognition")
    st.markdown("---")
    st.markdown(f"**Dataset (image):** `dataset(image)/`")
    st.markdown(f"**Dataset (CSV):** `Dataset.csv`")
    st.markdown(f"**Threshold:** `{RECOGNITION_THRESHOLD}`")
    st.markdown(f"**GPU:** {'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ CPU only'}")
    st.markdown("---")

    reg_st = registration_status()
    if reg_st["registered"]:
        st.markdown("**Registration:** 🟢 Active")
        st.markdown(f"People: **{len(reg_st['people'])}**")
        st.markdown(f"Embeddings: **{reg_st['total_embeddings']}**")
        for p in reg_st["people"]:
            st.markdown(f"  · {p}")
    elif reg_st.get("dataset_ready"):
        st.markdown("**Registration:** 🟡 Dataset ready")
        st.markdown("*Build the database in **Register Faces**.*")
        st.markdown(f"People found: **{len(reg_st['people'])}**")
    else:
        st.markdown("**Registration:** 🔴 No dataset found")
        st.markdown("*Add images under `dataset(image)/`.*")

    st.markdown("---")
    st.caption("DeepFace · Facenet512 · ONNX · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Overview",
    "🔍 Inspect & Preprocess",
    "📝 Register Faces",
    "📸 Live Webcam",
    "🎯 Recognize (Upload)",
    "⚡ Inference Benchmark",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="btn_refresh"):
        st.session_state.pop("img_summary", None)
        st.session_state.pop("csv_summary", None)

    if "img_summary" not in st.session_state:
        with st.spinner("Inspecting image dataset..."):
            prog = st.progress(0)
            msg_ph = st.empty()
            def update_prog(p, m):
                prog.progress(p); msg_ph.caption(m)
            st.session_state["img_summary"] = inspect_image_dataset(progress_callback=update_prog)
            prog.empty(); msg_ph.empty()

    if "csv_summary" not in st.session_state:
        with st.spinner("Parsing CSV dataset..."):
            st.session_state["csv_summary"] = inspect_csv_dataset()

    img  = st.session_state["img_summary"]
    csv  = st.session_state["csv_summary"]
    r_st = registration_status()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("👥 People",       len(img["people"]))
    c2.metric("🖼️ Images",       img["total_images"])
    c3.metric("✅ Valid",         img["total_valid"])
    c4.metric("❌ Invalid",       img["total_invalid"])
    c5.metric("😊 Faces Found",   img["total_faces"])
    c6.metric("🌐 CSV Entries",   csv["total_entries"])

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 👤 Custom Dataset — Per Person")
        if img["per_person"]:
            rows = []
            for name, stats in img["per_person"].items():
                reg = r_st["per_person"].get(name, 0) if r_st["registered"] else 0
                rows.append({
                    "Person": name, "Total": stats["images"],
                    "Valid": stats["valid"], "Invalid": stats["invalid"],
                    "Faces": stats["faces_detected"], "Registered": reg,
                    "Status": "✅ Registered" if reg > 0 else "⚠️ Not Registered",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            fig = px.bar(df, x="Person", y=["Valid", "Invalid", "Faces"],
                         barmode="group",
                         color_discrete_map={"Valid":"#3fb950","Invalid":"#f85149","Faces":"#58a6ff"},
                         template="plotly_dark", title="Images per Person")
            fig.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                               margin=dict(t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No persons found in dataset(image)/ folder.")

    with col_right:
        st.markdown("### 🌐 CSV Dataset — Per Label")
        if csv["per_label"]:
            csv_df = pd.DataFrame([
                {"Label": k, "Images": v}
                for k, v in sorted(csv["per_label"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(csv_df, use_container_width=True, hide_index=True, height=280)
            top15 = csv_df.head(15)
            fig2 = px.bar(top15, y="Label", x="Images", orientation="h",
                          color="Images", color_continuous_scale="Blues",
                          template="plotly_dark", title="Top 15 Labels by Image Count")
            fig2.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                                margin=dict(t=60), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Could not parse Dataset.csv.")

    if img["invalid_files"]:
        st.markdown("### ⚠️ Invalid Files")
        st.dataframe(pd.DataFrame(img["invalid_files"], columns=["File", "Reason"]),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Inspect & Preprocess
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 Inspect & Preprocess Dataset</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### 📂 Raw Image Inspection")
        if "img_summary" in st.session_state:
            img = st.session_state["img_summary"]
            for person, stats in img["per_person"].items():
                with st.expander(f"👤 {person}  —  {stats['images']} images, {stats['faces_detected']} faces"):
                    for f_info in stats["files"]:
                        status_icon = "✅" if f_info["face_detected"] else ("❌" if not f_info["valid"] else "⚠️")
                        label = (f_info["reason"] if f_info["reason"]
                                 else "Face detected" if f_info["face_detected"] else "No face found")
                        st.markdown(f"{status_icon} `{f_info['name']}` — {f_info['size_kb']} KB — {label}")
                        if f_info["valid"]:
                            try:
                                st.image(Image.open(f_info["path"]), width=160)
                            except Exception:
                                pass
        else:
            st.info("Run **Dataset Overview** tab first to load inspection data.")

    with col_b:
        st.markdown("#### ⚙️ Preprocessing")
        st.markdown("""
        Preprocessing will:
        - Detect faces in all images
        - Crop & align face regions (DeepFace)
        - Resize to **160×160**
        - Normalize pixel values
        - Save to `processed/` *(originals unchanged)*
        """)
        if st.button("🚀 Run Preprocessing", key="btn_preprocess"):
            with st.spinner("Preprocessing..."):
                prog2 = st.progress(0); msg2 = st.empty()
                def pp_cb(p, m): prog2.progress(p); msg2.caption(m)
                result = preprocess_dataset(progress_callback=pp_cb)
                prog2.empty(); msg2.empty()
                st.session_state["preprocess_result"] = result

        if "preprocess_result" in st.session_state:
            r = st.session_state["preprocess_result"]
            s = r["summary"]
            st.success(f"✅ Done — {s['succeeded']}/{s['total']} succeeded, {s['failed']} failed")
            st.markdown("#### 🖼️ Processed Thumbnails")
            for person, entries in r["processed"].items():
                ok_entries = [e for e in entries if e["success"]]
                if ok_entries:
                    st.markdown(f"**{person}**")
                    cols = st.columns(min(len(ok_entries), 6))
                    for ci, entry in enumerate(ok_entries[:6]):
                        with cols[ci]:
                            try:
                                st.image(Image.open(entry["dest"]), width=120,
                                         caption=os.path.basename(entry["dest"]))
                            except Exception:
                                pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Register Faces
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📝 Register Faces into Database</div>', unsafe_allow_html=True)
    r_st = registration_status()
    col1, col2 = st.columns([1, 1])

    with col1:
        if r_st["registered"]:
            st.success(f"✅ Database active — {len(r_st['people'])} people, {r_st['total_embeddings']} embeddings")
            st.markdown("**Registered people:**")
            for p in r_st["people"]:
                cnt = r_st["per_person"].get(p, 0)
                st.markdown(f"  - **{p}** — {cnt} embedding{'s' if cnt != 1 else ''}")
        elif r_st.get("dataset_ready"):
            st.info(f"Dataset ready: {len(r_st['people'])} people found. Build the embedding database below.")
            st.markdown("**People in the custom dataset:**")
            for person in r_st["people"]:
                st.markdown(f"- **{person}**")
        else:
            st.warning("No valid people folders found under `dataset(image)/`.")

        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Images split 70% → registration, 30% → testing
        2. DeepFace (Facenet512) generates 512-dim embeddings
        3. Saved to `embeddings_db/embeddings.pkl`
        """)

        if st.button("🔐 Register All Faces", key="btn_register"):
            with st.spinner("Registering faces... (first run downloads Facenet512 model ~90MB)"):
                prog3 = st.progress(0); msg3 = st.empty()
                def reg_cb(p, m): prog3.progress(p); msg3.caption(m)
                reg_result = build_registration_database(progress_callback=reg_cb)
                prog3.empty(); msg3.empty()
                if "error" in reg_result:
                    st.error(reg_result["error"])
                else:
                    st.session_state["reg_result"] = reg_result
                    st.rerun()

    with col2:
        if "reg_result" in st.session_state or r_st["registered"]:
            if "reg_result" in st.session_state:
                reg_res = st.session_state["reg_result"]
                s = reg_res["summary"]
                st.markdown("### 📊 Registration Summary")
                s_c1, s_c2, s_c3 = st.columns(3)
                s_c1.metric("👥 People", s["people"])
                s_c2.metric("✅ Registered", s["registered"])
                s_c3.metric("❌ Failed", s["failed"])

                st.markdown("### 🔪 Train / Test Split")
                split = reg_res["split_info"]
                split_rows = [{"Person": p, "Register": len(i["register"]),
                                "Test": len(i["test"]),
                                "Total": len(i["register"]) + len(i["test"])}
                               for p, i in split.items()]
                st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

                db = reg_res["registration_db"]
                if db:
                    pie_df = pd.DataFrame([{"Person": p, "Embeddings": len(e)} for p, e in db.items()])
                    fig_pie = px.pie(pie_df, values="Embeddings", names="Person",
                                     template="plotly_dark", hole=0.4,
                                     title="Embedding Distribution",
                                     color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig_pie.update_layout(paper_bgcolor="#0d1117", margin=dict(t=60))
                    st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            if st.button("🧪 Evaluate on Test Set", key="btn_eval"):
                with st.spinner("Running evaluation..."):
                    db, split = load_registration_database()
                    eval_res = evaluate_on_test_set(registration_db=db, split_info=split)
                    st.session_state["eval_result"] = eval_res

            if "eval_result" in st.session_state:
                ev = st.session_state["eval_result"]
                st.markdown(f"### 🎯 Test Accuracy: **{ev['accuracy']*100:.1f}%** ({ev['correct']}/{ev['total']})")
                if ev["per_person"]:
                    per_df = pd.DataFrame([
                        {"Person": p, "Correct": s["correct"], "Total": s["total"],
                         "Accuracy": f"{s['accuracy']*100:.0f}%"}
                        for p, s in ev["per_person"].items()
                    ])
                    st.dataframe(per_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE WEBCAM RECOGNITION  (★ main new feature)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📸 Live Webcam Face Recognition</div>',
                unsafe_allow_html=True)

    r_st4 = registration_status()

    if not r_st4["registered"]:
        st.error("⚠️ No registration database found. Please go to **Register Faces** tab first.")
    else:
        # Load DB once into session state
        if "webcam_db" not in st.session_state:
            db_tmp, _ = load_registration_database()
            st.session_state["webcam_db"] = db_tmp

        db_loaded = st.session_state["webcam_db"]

        # Controls row
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
        with ctrl1:
            threshold_wc = st.slider(
                "Threshold", min_value=0.20, max_value=0.90,
                value=float(RECOGNITION_THRESHOLD), step=0.01, key="wc_threshold"
            )
        with ctrl2:
            show_perf = st.checkbox("Show inference timing", value=True, key="wc_perf")
        with ctrl3:
            st.info("📷 Take a webcam photo to identify the face and view the active inference backend and latency.")

        st.markdown("---")

        # Webcam capture
        cam_col, result_col = st.columns([1, 1])

        with cam_col:
            st.markdown("### 📷 Camera")
            cam_image = st.camera_input("Take a photo to identify")

        with result_col:
            st.markdown("### 🔎 Result")

            if cam_image is not None:
                # Load image
                query_pil = Image.open(cam_image).convert("RGB")
                query_np  = np.array(query_pil)

                with st.spinner("🧠 Analyzing face..."):
                    result, inference_ms, inference_device = recognize_with_timing(
                        query_np, db_loaded, threshold_wc
                    )

                identity   = result["identity"]
                confidence = result["confidence"]
                dist       = result["distance"]

                # Annotate image with bounding box + name
                annotated = annotate_image(query_np, identity, confidence,
                                           face_location=result.get("face_location"))

                # Show annotated image
                st.image(annotated, caption="Detected Face", use_container_width=True)

                # ── Identity badge ─────────────────────────────────────────
                if identity != "Unknown" and result["success"]:
                    st.markdown(f"""
                    <div class="result-card-match">
                        <div style="font-size:1rem; color:#8b949e;">IDENTIFIED AS</div>
                        <div class="name-display">✅ {identity.upper()}</div>
                        <div style="color:#3fb950; font-size:1.1rem;">
                            Confidence: <strong>{confidence*100:.1f}%</strong>
                        </div>
                        <div style="color:#8b949e; font-size:0.85rem; margin-top:6px;">
                            Distance: {dist:.4f} &nbsp;|&nbsp; Threshold: {threshold_wc}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    reason = result.get("reason", "Face not in database")
                    st.markdown(f"""
                    <div class="result-card-unknown">
                        <div style="font-size:1rem; color:#8b949e;">RESULT</div>
                        <div class="name-display-unknown">❓ UNKNOWN</div>
                        <div style="color:#f85149; font-size:0.9rem;">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Active inference backend metrics ───────────────────────────────
        if cam_image is not None and show_perf:
            st.markdown("---")
            st.markdown("### ⚡ Inference Performance")
            perf1, perf2 = st.columns(2)
            perf1.metric("Active backend", inference_device)
            perf2.metric("Recognition latency", f"{inference_ms:.1f} ms")

        # ── History of last 10 captures ────────────────────────────────────
        if cam_image is not None:
            if "webcam_history" not in st.session_state:
                st.session_state["webcam_history"] = []

            st.session_state["webcam_history"].insert(0, {
                "Identity":  identity,
                "Confidence": f"{confidence*100:.1f}%",
                "Backend": inference_device,
                "Latency (ms)": f"{inference_ms:.1f}",
                "Time":      time.strftime("%H:%M:%S"),
            })
            st.session_state["webcam_history"] = st.session_state["webcam_history"][:10]

        if "webcam_history" in st.session_state and st.session_state["webcam_history"]:
            st.markdown("---")
            st.markdown("### 📋 Recognition History (last 10)")
            hist_df = pd.DataFrame(st.session_state["webcam_history"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Clear History", key="btn_clear_hist"):
                st.session_state["webcam_history"] = []
                st.rerun()

        # Quick links to registered faces
        if r_st4["registered"] and len(r_st4["people"]) > 0:
            st.markdown("---")
            st.markdown("### 👥 Registered Identities")
            n_people = max(1, len(r_st4["people"]))
            people_cols = st.columns(n_people)
            for ci, person in enumerate(r_st4["people"]):
                with people_cols[ci]:
                    # Show one sample image per person
                    pdir = os.path.join(IMAGE_DATASET_DIR, person)
                    if os.path.isdir(pdir):
                        imgs = [f for f in os.listdir(pdir)
                                if f.lower().endswith((".jpg",".jpeg",".png"))]
                        if imgs:
                            try:
                                pil_s = Image.open(os.path.join(pdir, imgs[0]))
                                st.image(pil_s, caption=person, use_container_width=True)
                            except Exception:
                                st.caption(person)
                        else:
                            st.caption(person)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Recognize Face (Upload)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🎯 Recognize a Face (Upload)</div>', unsafe_allow_html=True)

    r_st5 = registration_status()
    if not r_st5["registered"]:
        st.error("⚠️ No registration database found. Please go to **Register Faces** tab first.")
    else:
        col_upload, col_result = st.columns([1, 1])

        with col_upload:
            st.markdown("#### 📤 Upload Image")
            uploaded = st.file_uploader("Drop an image here",
                                        type=["jpg","jpeg","png","bmp"],
                                        key="recog_upload")
            st.markdown("**— or use a test image —**")
            _, split_info = load_registration_database()
            test_images = []
            if split_info:
                for person, info in split_info.items():
                    for fp in info.get("test", []):
                        test_images.append((person, fp))

            if test_images:
                test_options = [f"{os.path.basename(fp)} ({p})" for p, fp in test_images]
                selected_test = st.selectbox("Select test image:", ["— choose —"] + test_options)
            else:
                selected_test = "— choose —"
                st.caption("No test images available.")

            threshold_up = st.slider("Threshold", min_value=0.2, max_value=0.9,
                                      value=float(RECOGNITION_THRESHOLD), step=0.01,
                                      key="upload_threshold")

            query_img_np = None
            if uploaded:
                query_pil    = Image.open(uploaded).convert("RGB")
                query_img_np = np.array(query_pil)
                st.image(query_pil, caption="Uploaded Image", use_container_width=True)
            elif selected_test != "— choose —":
                idx = test_options.index(selected_test)
                _, fp = test_images[idx]
                query_pil    = Image.open(fp).convert("RGB")
                query_img_np = np.array(query_pil)
                st.image(query_pil, caption=f"Test: {os.path.basename(fp)}", use_container_width=True)

        with col_result:
            st.markdown("#### 🔎 Recognition Result")
            if query_img_np is not None and st.button("🔍 Identify + Time", key="btn_identify"):
                with st.spinner("Analyzing..."):
                    db, _ = load_registration_database()
                    result, inference_ms, inference_device = recognize_with_timing(
                        query_img_np, db, threshold_up
                    )

                identity   = result["identity"]
                confidence = result["confidence"]
                dist       = result["distance"]

                # Annotated image
                annotated = annotate_image(
                    query_img_np,
                    identity,
                    confidence,
                    face_location=result.get("face_location"),
                )
                st.image(annotated, use_container_width=True)

                if result["success"] and identity != "Unknown":
                    st.success(f"✅ **{identity}** — {confidence*100:.1f}% confidence")
                else:
                    st.error(f"❓ Unknown — {result.get('reason','')}")

                st.markdown(f"**Distance:** `{dist:.4f}` | **Threshold:** `{threshold_up}`")

                # Performance
                st.markdown("---")
                st.markdown("**⚡ Inference Times:**")
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Inference backend", inference_device)
                pm2.metric("Recognition latency", f"{inference_ms:.1f} ms")
                pm3.metric("GPU acceleration", "Active" if GPU_AVAILABLE else "Unavailable")

                # Distance bar chart
                if result["all_distances"]:
                    dist_df = pd.DataFrame([
                        {"Person": p, "Distance": d, "Match": p == identity}
                        for p, d in sorted(result["all_distances"].items(), key=lambda x: x[1])
                    ])
                    fig_dist = px.bar(dist_df, x="Person", y="Distance",
                                      color="Match",
                                      color_discrete_map={True: "#3fb950", False: "#58a6ff"},
                                      template="plotly_dark",
                                      title="Distance to Each Registered Person")
                    fig_dist.add_hline(y=threshold_up, line_dash="dash",
                                       line_color="#f85149",
                                       annotation_text=f"Threshold ({threshold_up})")
                    fig_dist.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Active-provider inference benchmark
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">⚡ Inference Benchmark</div>', unsafe_allow_html=True)

    col_info, col_run = st.columns([2, 1])
    with col_info:
        st.markdown(f"""
        **Hardware:**
        - GPU: `{'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ Not available — CPU-only results shown'}`

        **CPU vs GPU on the same detected face crops, batch sizes (1, 4, 8, 16):**
        - Inference time per image (ms)
        - Throughput (images/second)
        - Memory usage (MB RSS)
        - Host CPU utilization %

        *Face detection runs once before timing. The comparison measures Facenet512 embedding inference only.*
        """)

    with col_run:
        st.markdown("&nbsp;")
        if st.button("🚀 Run CPU + GPU Benchmark", key="btn_benchmark"):
            with st.spinner("Benchmarking (may take a few minutes)..."):
                prog5 = st.progress(0); msg5 = st.empty()
                def bm_cb(p, m): prog5.progress(p); msg5.caption(m)
                bm_result = run_benchmark(progress_callback=bm_cb)
                prog5.empty(); msg5.empty()
                st.session_state["bm_result"] = bm_result

    if "bm_result" in st.session_state:
        bm = st.session_state["bm_result"]
        st.markdown(f"**Images used:** `{bm['images_available']}`")
        st.markdown(f"**Faces detected:** `{bm.get('faces_available', 0)}`")

        if not bm["gpu_available"]:
            st.warning("No supported GPU inference provider is available. Only CPU results are shown.")
        for device, error in bm.get("errors", {}).items():
            st.error(f"{device} benchmark could not run: {error}")

        results = bm["results"]
        if not results:
            st.warning("No benchmark results. Ensure dataset images exist.")
        else:
            df_bm = pd.DataFrame(results)

            cpu_rows = df_bm[df_bm["device"] == "CPU"]
            gpu_rows = df_bm[df_bm["device"] == "GPU"]
            cpu_ms = cpu_rows["avg_time_per_image_ms"].min() if not cpu_rows.empty else None
            gpu_ms = gpu_rows["avg_time_per_image_ms"].min() if not gpu_rows.empty else None
            gpu_tp = gpu_rows["throughput_imgs_sec"].max() if not gpu_rows.empty else None
            speedups = list(bm.get("speedup_by_batch", {}).values())
            mean_speedup = float(np.mean(speedups)) if speedups else None

            # ── CPU/GPU summary ────────────────────────────────────────────
            st.markdown("---")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Best CPU latency", f"{cpu_ms:.1f} ms/image" if cpu_ms is not None else "N/A")
            sm2.metric("Best GPU latency", f"{gpu_ms:.1f} ms/image" if gpu_ms is not None else "N/A")
            sm3.metric("Best GPU throughput", f"{gpu_tp:.2f} images/s" if gpu_tp is not None else "N/A")
            sm4.metric("Mean GPU speedup", f"{mean_speedup:.2f}×" if mean_speedup is not None else "N/A")

            st.markdown("### 📋 Raw Results")
            display_df = df_bm[["device","batch_size","avg_time_per_image_ms",
                                 "throughput_imgs_sec","memory_mb","cpu_pct","gpu_pct"]].rename(columns={
                "device":"Device","batch_size":"Batch","avg_time_per_image_ms":"ms/img",
                "throughput_imgs_sec":"img/s","memory_mb":"RAM MB","cpu_pct":"CPU %","gpu_pct":"GPU %"
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ── Charts ────────────────────────────────────────────────────
            col_c1, col_c2 = st.columns(2)

            with col_c1:
                fig_t = px.line(df_bm, x="batch_size", y="avg_time_per_image_ms",
                                 color="device", markers=True,
                                 color_discrete_map={"CPU":"#58a6ff","GPU":"#3fb950"},
                                 template="plotly_dark",
                                 title="⏱️ Inference Time per Image (ms)",
                                 labels={"batch_size":"Batch","avg_time_per_image_ms":"ms/img"})
                fig_t.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                st.plotly_chart(fig_t, use_container_width=True)

                fig_m = px.bar(df_bm, x="batch_size", y="memory_mb", color="device",
                                barmode="group",
                                color_discrete_map={"CPU":"#58a6ff","GPU":"#3fb950"},
                                template="plotly_dark", title="💾 Memory Usage (MB)",
                                labels={"batch_size":"Batch","memory_mb":"RAM MB"})
                fig_m.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                st.plotly_chart(fig_m, use_container_width=True)

            with col_c2:
                fig_tp = px.bar(df_bm, x="batch_size", y="throughput_imgs_sec",
                                 color="device", barmode="group",
                                 color_discrete_map={"CPU":"#58a6ff","GPU":"#3fb950"},
                                 template="plotly_dark", title="🚀 Throughput (img/s)",
                                 labels={"batch_size":"Batch","throughput_imgs_sec":"img/s"})
                fig_tp.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                st.plotly_chart(fig_tp, use_container_width=True)

                speedups = bm.get("speedup_by_batch", {})
                if speedups:
                    speed_df = pd.DataFrame([
                        {"Batch": batch, "GPU speedup (×)": value}
                        for batch, value in speedups.items()
                    ])
                    fig_speed = px.bar(
                        speed_df,
                        x="Batch",
                        y="GPU speedup (×)",
                        template="plotly_dark",
                        color_discrete_sequence=["#d29922"],
                        title="⚡ GPU speedup over CPU",
                    )
                    fig_speed.add_hline(y=1, line_dash="dash", line_color="#8b949e")
                    fig_speed.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                    st.plotly_chart(fig_speed, use_container_width=True)
                else:
                    st.info("GPU speedup is unavailable because a paired CPU/GPU batch did not complete.")

            # Host CPU utilization
            cpu_rows2 = df_bm[["batch_size", "cpu_pct"]].dropna()
            if not cpu_rows2.empty:
                fig_cpu = px.line(cpu_rows2, x="batch_size", y="cpu_pct", markers=True,
                                   template="plotly_dark", color_discrete_sequence=["#58a6ff"],
                                   title="🖥️ Host CPU Utilization During Inference",
                                   labels={"batch_size":"Batch","cpu_pct":"CPU %"})
                fig_cpu.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                st.plotly_chart(fig_cpu, use_container_width=True)
