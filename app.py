# app.py — Streamlit Face Recognition System
# Sabhi 3 steps ek jagah: Dataset → Train → Recognize
# Run karo: streamlit run app.py

import streamlit as st
import cv2
import numpy as np
import os
import json
import time
import shutil
from PIL import Image

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG  ← sabse pehle hona chahiye
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"] { background: #0d0d1a; }
[data-testid="stSidebar"]          { background: #111122; border-right: 1px solid #252540; }
section.main > div                 { padding-top: 0.8rem; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600   !important;
    color: #9999bb     !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #4a2f8a !important;
    color: #ffffff      !important;
}

/* ── Generic card ── */
.card {
    background: #181830;
    border: 1px solid #252545;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #e84560;
    margin-bottom: 0.5rem;
}

/* ── Metric pair ── */
.metric-row  { display:flex; gap:0.8rem; margin:0.6rem 0; }
.metric-box  {
    flex:1; background:#13132a; border:1px solid #252545;
    border-radius:12px; padding:0.8rem; text-align:center;
}
.metric-box .val { font-size:1.9rem; font-weight:800; color:#e84560; line-height:1; }
.metric-box .lbl { font-size:0.7rem; color:#888aaa; margin-top:0.2rem;
                   text-transform:uppercase; letter-spacing:0.8px; }

/* ── Status badges ── */
.badge-ok  { background:#0d2b1a; border:1px solid #00c853; border-radius:8px;
             padding:0.45rem 0.9rem; color:#00e676; font-size:0.82rem; font-weight:600; }
.badge-warn{ background:#2b1c0d; border:1px solid #ff9800; border-radius:8px;
             padding:0.45rem 0.9rem; color:#ffb74d; font-size:0.82rem; font-weight:600; }

/* ── Result boxes ── */
.result-known {
    background:#0d2b1a; border:2px solid #00c853; border-radius:12px;
    padding:1rem 1.2rem; color:#00e676; font-size:1.05rem;
    font-weight:700; text-align:center; margin-bottom:0.6rem;
}
.result-unknown {
    background:#2b0d0d; border:2px solid #e53935; border-radius:12px;
    padding:1rem 1.2rem; color:#ff5252; font-size:1.05rem;
    font-weight:700; text-align:center; margin-bottom:0.6rem;
}
.result-noface {
    background:#1a1a2e; border:1px dashed #444466; border-radius:12px;
    padding:1.5rem; color:#777799; font-size:0.9rem; text-align:center;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg,#1a1a35 0%,#161630 50%,#0f2d55 100%);
    border:1px solid #4a2f8a; border-radius:16px;
    padding:1.4rem 2rem; margin-bottom:1.2rem;
    display:flex; align-items:center; gap:1rem;
}
.header-banner h1 { color:#e84560; font-size:1.9rem; font-weight:800;
                    margin:0; letter-spacing:-0.5px; }
.header-banner p  { color:#888aaa; margin:0; font-size:0.88rem; }

/* ── Step arrows ── */
.steps { display:flex; align-items:center; gap:0.4rem; margin-bottom:1rem; }
.step  {
    flex:1; background:#181830; border:1px solid #252545;
    border-radius:10px; padding:0.6rem 0.5rem;
    text-align:center; font-size:0.8rem; color:#888aaa;
}
.step b { display:block; color:#cccce0; margin-top:0.1rem; }
.arrow  { color:#4a2f8a; font-size:1.3rem; }

/* ── Sidebar title ── */
.sb-title { color:#e84560; font-size:1.1rem; font-weight:800;
            letter-spacing:0.5px; margin-bottom:0.6rem; }

/* ── Person card in sidebar ── */
.person-card {
    background:#13132a; border:1px solid #252545; border-radius:10px;
    padding:0.55rem 0.85rem; margin:0.3rem 0;
    display:flex; justify-content:space-between; align-items:center;
}
.person-card .pn { color:#ccccdd; font-weight:600; font-size:0.87rem; }
.person-card .pc { color:#e84560; font-weight:700; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
DATASET_PATH         = "dataset"
FACE_SIZE            = (100, 100)
MODEL_PATH           = "face_model.yml"
LABELS_PATH          = "labels.json"
CONFIDENCE_THRESHOLD = 70          # LBPH: lower = better match
MAX_IMAGES_DEFAULT   = 30

os.makedirs(DATASET_PATH, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for key, default in [("model_version", 0), ("cam_key", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════
#  HAAR CASCADE — Windows / Linux / Mac safe path finder
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_cascade():
    filename = "haarcascade_frontalface_default.xml"
    candidates = []

    # 1. cv2.data (modern opencv-python)
    try:
        candidates.append(cv2.data.haarcascades + filename)
    except AttributeError:
        pass

    # 2. cv2 package folder → data subfolder  (Anaconda Windows)
    try:
        cv2_pkg = os.path.dirname(os.path.abspath(cv2.__file__))
        # On Anaconda: cv2.__file__ = .../site-packages/cv2/cv2.pyd
        # haarcascades are in  .../site-packages/cv2/data/
        candidates.append(os.path.join(cv2_pkg, "data", filename))
        # Sometimes cv2.__file__ points to the .pyd directly inside cv2/
        candidates.append(os.path.join(os.path.dirname(cv2_pkg), "cv2", "data", filename))
    except Exception:
        pass

    # 3. importlib origin (catches edge cases)
    try:
        import importlib.util
        spec = importlib.util.find_spec("cv2")
        if spec and spec.origin:
            origin_dir = os.path.dirname(os.path.abspath(spec.origin))
            candidates.append(os.path.join(origin_dir, "data", filename))
    except Exception:
        pass

    # 4. Common system paths
    candidates += [
        f"/usr/share/opencv4/haarcascades/{filename}",
        f"/usr/share/opencv/haarcascades/{filename}",
        f"/usr/local/share/opencv4/haarcascades/{filename}",
        r"C:\opencv\build\etc\haarcascades\\" + filename,
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return cv2.CascadeClassifier(path)

    # Could not find — show helpful error
    tried = "\n".join(f"  • {p}" for p in candidates if p)
    st.error(
        "❌ **Haar Cascade XML file not found!**\n\n"
        "**Fix (run in terminal):**\n"
        "```\npip install opencv-contrib-python --upgrade --force-reinstall\n```\n\n"
        f"**Paths checked:**\n```\n{tried}\n```"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════
#  RECOGNIZER LOADER  (cache version-aware)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_recognizer(_version: int):
    """_version changes → cache miss → fresh load."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, {}
    try:
        try:
            rec = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            return None, {"__error__": "contrib"}
        rec.read(MODEL_PATH)
        with open(LABELS_PATH) as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
        return rec, label_map
    except Exception:
        return None, {}


# ══════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════
face_cascade = load_cascade()


def get_dataset_info() -> dict:
    """{ person_name: [img1, img2, …] }"""
    info = {}
    if not os.path.isdir(DATASET_PATH):
        return info
    for person in sorted(os.listdir(DATASET_PATH)):
        p = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(p):
            continue
        imgs = sorted(
            f for f in os.listdir(p)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if imgs:
            info[person] = imgs
    return info


def count_imgs(person: str) -> int:
    p = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(p):
        return 0
    return len([f for f in os.listdir(p) if f.lower().endswith(".jpg")])


def detect_faces(img_bgr):
    """Return list of (x,y,w,h) from BGR image."""
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq  = cv2.equalizeHist(gray)
    gray_bl  = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    return face_cascade.detectMultiScale(
        gray_bl, scaleFactor=1.05, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )


def camera_bytes_to_bgr(cam_file) -> np.ndarray:
    arr = np.frombuffer(cam_file.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_pil(img_bgr) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_open_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    return img  # BGR


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sb-title">📁 Dataset Overview</div>',
                    unsafe_allow_html=True)

        info = get_dataset_info()

        if not info:
            st.markdown("""
            <div style="color:#777799;font-size:0.84rem;background:#13132a;
                 border:1px solid #252545;border-radius:10px;padding:0.9rem;margin-top:0.4rem;">
                 Dataset is empty.<br>Capture faces in Tab 1 first.
            </div>""", unsafe_allow_html=True)
        else:
            total_p = len(info)
            total_i = sum(len(v) for v in info.values())
            model_ok = os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)

            # Metrics
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="val">{total_p}</div>
                    <div class="lbl">Persons</div>
                </div>
                <div class="metric-box">
                    <div class="val">{total_i}</div>
                    <div class="lbl">Images</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Model status
            if model_ok:
                st.markdown('<div class="badge-ok">✅ Model Trained & Ready</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-warn">⚠️ Model Not Trained</div>',
                            unsafe_allow_html=True)

            st.divider()

            # Per-person expandable with thumbnails + individual delete
            for person, imgs in info.items():
                with st.expander(f"👤  {person}  ·  {len(imgs)} photos"):
                    person_path = os.path.join(DATASET_PATH, person)

                    # Show all images with delete button per image (3 cols)
                    cols = st.columns(3)
                    deleted_any = False
                    for i, img_name in enumerate(imgs):
                        img_full_path = os.path.join(person_path, img_name)
                        col = cols[i % 3]
                        try:
                            pil_img = Image.open(img_full_path)
                            col.image(pil_img, width='stretch',
                                      caption=img_name)
                        except Exception:
                            col.markdown("⚠️ Error")
                        # Delete button for each image
                        if col.button("🗑️", key=f"del_img_{person}_{img_name}",
                                      help=f"Delete {img_name}", use_container_width=True):
                            try:
                                os.remove(img_full_path)
                                deleted_any = True
                            except Exception:
                                pass

                    if deleted_any:
                        st.rerun()

                    st.markdown("---")
                    # Delete entire person folder
                    if st.button(f"🗑️ Delete All — {person}",
                                 key=f"del_person_{person}",
                                 type="secondary", use_container_width=True):
                        shutil.rmtree(person_path, ignore_errors=True)
                        st.success(f"'{person}' and all their images have been deleted!")
                        st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 1 — DATASET CREATE
# ══════════════════════════════════════════════════════════════
def tab_dataset():
    st.markdown("""
    <div class="card">
        <div class="card-title">📸 Step 1 — Dataset Create</div>
        <span style="color:#888aaa;font-size:0.87rem;">
            Enter your name → Take a photo → Face gets detected → Save it.<br>
            Run multiple times for the same person to capture more photos.
        </span>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ── Left: Settings ────────────────────────────────────────
    with left:
        st.markdown("#### 👤 Person Details")

        person_name = st.text_input(
            "Enter your name",
            placeholder="e.g. Rahul, Priya, Nisha …",
            key="ds_name"
        )
        max_imgs = st.slider("Target images", 5, 100, MAX_IMAGES_DEFAULT, 5)

        if person_name:
            name_clean = person_name.strip().title()
            existing   = count_imgs(name_clean)
            pct        = min(existing / max(max_imgs, 1), 1.0)

            st.markdown(f"""
            <div class="card" style="margin-top:0.6rem;">
                <span style="color:#ccccdd;font-size:0.9rem;">
                    <b style="color:#e84560;">{name_clean}</b> — Status
                </span><br>
                <span style="color:#888aaa;font-size:0.82rem;">
                    Captured: <b style="color:#fff;">{existing}</b> /
                    <b style="color:#fff;">{max_imgs}</b> images
                </span>
            </div>""", unsafe_allow_html=True)
            st.progress(pct)

            if existing == 0:
                st.info("💡 First time — try different angles for better accuracy!")
            elif existing < 15:
                st.warning("⚠️ 20–30 images recommended for better accuracy.")
            else:
                st.success("✅ Great progress! Recognition will be more accurate.")
        else:
            st.markdown("""
            <div style="color:#777799;font-size:0.85rem;background:#13132a;
                 border:1px dashed #252545;border-radius:10px;padding:1rem;
                 margin-top:1rem;text-align:center;">
                 ⬆️ Please enter your name first
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="color:#777799;font-size:0.8rem;line-height:1.75;">
            <b style="color:#cccce0;">📌 Tips for better accuracy:</b><br>
            • Look straight at the camera<br>
            • Try slight left, right and tilt angles<br>
            • Capture in different lighting conditions<br>
            • With and without glasses — both<br>
            • Aim for at least 20 images
        </div>""", unsafe_allow_html=True)

    # ── Right: Camera ─────────────────────────────────────────
    with right:
        st.markdown("#### 📷 Camera Capture")

        if not person_name:
            st.markdown("""
            <div style="background:#13132a;border:1px dashed #252545;border-radius:12px;
                 padding:3.5rem;text-align:center;color:#777799;">
                 👈 Enter your name first
            </div>""", unsafe_allow_html=True)
            return

        name_clean = person_name.strip().title()

        cam_img = st.camera_input(
            "📷 Take a photo — then click Save Face",
            key=f"cam_{st.session_state.cam_key}",
        )

        if cam_img:
            img_bgr = camera_bytes_to_bgr(cam_img)
            faces   = detect_faces(img_bgr)
            preview = img_bgr.copy()

            if len(faces) == 0:
                # Red guide box
                h_i, w_i = img_bgr.shape[:2]
                cx, cy, bw, bh = w_i // 2, h_i // 2, 180, 200
                cv2.rectangle(preview,
                    (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2),
                    (0, 0, 210), 2)
                cv2.putText(preview, "No Face Found",
                    (cx - 80, cy + bh//2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 210), 2)
                st.image(bgr_to_pil(preview), width='stretch')
                st.error("❌ No face detected — move closer with good lighting!")

            else:
                (x, y, w, h) = faces[0]
                cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 210, 80), 2)
                cv2.putText(preview, "✓ Face Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 80), 2)

                crop_preview = cv2.resize(img_bgr[y:y+h, x:x+w], (110, 110))
                c1, c2 = st.columns([3, 1])
                c1.image(bgr_to_pil(preview), width='stretch')
                c2.markdown("<br>", unsafe_allow_html=True)
                c2.image(bgr_to_pil(crop_preview), caption="Crop",
                         width='stretch')

                st.success("✅ Face detected! Click Save to proceed.")

                col_save, col_skip = st.columns(2)
                if col_save.button("💾 Save Face", type="primary", use_container_width=True):
                    person_path = os.path.join(DATASET_PATH, name_clean)
                    os.makedirs(person_path, exist_ok=True)
                    nxt = count_imgs(name_clean) + 1

                    # Save COLOR image
                    face_color   = img_bgr[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_color, FACE_SIZE,
                                             interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(person_path, f"{nxt}.jpg"), face_resized)

                    st.session_state.cam_key += 1
                    st.success(f"✅ Image #{nxt} saved for **{name_clean}**!")
                    time.sleep(0.4)
                    st.rerun()

                if col_skip.button("🔄 Retake", use_container_width=True):
                    st.session_state.cam_key += 1
                    st.rerun()


    # ── Manage Saved Images ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '''<div class="card">
        <div class="card-title">🖼️ Manage Saved Images</div>
        <span style="color:#888aaa;font-size:0.87rem;">
            View your saved face images and delete unwanted ones.
        </span>
        </div>''', unsafe_allow_html=True)

    info_mgr = get_dataset_info()
    if not info_mgr:
        st.markdown(
            '<div style="color:#777799;font-size:0.85rem;background:#13132a;' +
            'border:1px dashed #252545;border-radius:10px;padding:1rem;text-align:center;">' +
            'No images saved yet.</div>',
            unsafe_allow_html=True)
    else:
        person_list     = list(info_mgr.keys())
        selected_person = st.selectbox("Select Person:", person_list,
                                       key="manage_person_select")
        if selected_person:
            imgs        = info_mgr[selected_person]
            person_path = os.path.join(DATASET_PATH, selected_person)
            total_count = len(imgs)

            st.markdown(
                f'<div class="card" style="margin-bottom:0.8rem;">' +
                f'<span style="color:#ccccdd;font-size:0.9rem;">' +
                f'<b style="color:#e84560;">{selected_person}</b> — ' +
                f'<b style="color:#fff;">{total_count}</b> images saved</span></div>',
                unsafe_allow_html=True)

            if st.button(f"🗑️ Delete ALL {total_count} images of {selected_person}",
                         type="secondary", key="del_all_main"):
                shutil.rmtree(person_path, ignore_errors=True)
                st.success(f"'{selected_person}' — all {total_count} images deleted!")
                st.rerun()

            st.markdown("---")
            st.markdown(f"**{selected_person}** — images. Click 🗑️ to delete any image:")

            COLS = 4
            deleted_flag = False
            for row_start in range(0, len(imgs), COLS):
                row_imgs  = imgs[row_start: row_start + COLS]
                grid_cols = st.columns(COLS)
                for j, img_name in enumerate(row_imgs):
                    img_path = os.path.join(person_path, img_name)
                    col = grid_cols[j]
                    try:
                        col.image(Image.open(img_path), width='stretch')
                    except Exception:
                        col.markdown("⚠️ Error")
                    col.markdown(
                        f'<div style="text-align:center;color:#777799;' +
                        f'font-size:0.7rem;margin-top:-4px;">{img_name}</div>',
                        unsafe_allow_html=True)
                    if col.button("🗑️", key=f"del_{selected_person}_{img_name}",
                                  help=f"Delete {img_name}", use_container_width=True):
                        try:
                            os.remove(img_path)
                            deleted_flag = True
                        except Exception:
                            pass

            if deleted_flag:
                st.success("✅ Image deleted successfully!")
                time.sleep(0.3)
                st.rerun()



# ══════════════════════════════════════════════════════════════
#  TAB 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════
def tab_train():
    st.markdown("""
    <div class="card">
        <div class="card-title">🧠 Step 2 — Model Training</div>
        <span style="color:#888aaa;font-size:0.87rem;">
            Train all persons from the dataset at once.
            Re-train whenever a new person is added.
        </span>
    </div>""", unsafe_allow_html=True)

    info = get_dataset_info()

    if not info:
        st.markdown("""
        <div class="badge-warn" style="padding:1rem;">
            ❌ Dataset is empty! Capture faces in Tab 1 first.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown("#### 👥 Persons available for training:")
    total_imgs = 0
    has_warning = False

    for person, imgs in info.items():
        c = len(imgs)
        total_imgs += c
        pct = min(c / 30, 1.0)
        col1, col2, col3 = st.columns([3, 1, 2])
        col1.markdown(
            f"<span style='color:#ccccdd;font-weight:600;'>👤 {person}</span>",
            unsafe_allow_html=True)
        col2.markdown(
            f"<span style='color:#e84560;font-weight:700;'>{c} imgs</span>",
            unsafe_allow_html=True)
        col3.progress(pct)
        if c < 10:
            has_warning = True

    if has_warning:
        st.warning("⚠️ Some persons have fewer than 10 images — accuracy may be affected.")

    st.markdown(f"""
    <div class="metric-row" style="margin:0.8rem 0;">
        <div class="metric-box">
            <div class="val">{len(info)}</div><div class="lbl">Persons</div>
        </div>
        <div class="metric-box">
            <div class="val">{total_imgs}</div><div class="lbl">Total Images</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚀 Training Start Karo", type="primary", use_container_width=True):
        _do_training(info)


def _do_training(info: dict):
    prog    = st.progress(0.0, text="Starting training …")
    status  = st.empty()
    log_box = st.empty()
    logs    = []

    try:
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            st.error("""
❌ **opencv-contrib-python not found!**

`cv2.face` module is not available. Fix this by running:

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

Then restart Streamlit and try again.
""")
            return
        faces_list  = []
        labels_list = []
        label_map   = {}

        names = sorted(info.keys())
        n     = len(names)

        for idx, person in enumerate(names):
            status.info(f"🔄 Processing: **{person}** ({idx+1}/{n})")
            label_map[idx] = person
            person_path    = os.path.join(DATASET_PATH, person)
            loaded = 0

            for img_name in info[person]:
                img_path = os.path.join(person_path, img_name)
                # Load as grayscale — LBPH needs grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_eq      = cv2.equalizeHist(img)
                img_resized = cv2.resize(img_eq, FACE_SIZE,
                                         interpolation=cv2.INTER_CUBIC)
                faces_list.append(img_resized)
                labels_list.append(idx)
                loaded += 1

            logs.append(f"  [{idx}] '{person}' → {loaded} images loaded")
            log_box.code("\n".join(logs), language=None)
            prog.progress((idx + 1) / n,
                          text=f"Processing {idx+1}/{n} persons …")

        if not faces_list:
            st.error("❌ No images could be loaded! Please check your dataset.")
            return

        status.info(f"🔄 {len(faces_list)} faces se model train ho raha hai …")
        recognizer.train(faces_list, np.array(labels_list))
        recognizer.save(MODEL_PATH)

        with open(LABELS_PATH, "w") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

        prog.progress(1.0, text="✅ Training complete!")
        logs += ["", "✓ face_model.yml saved", "✓ labels.json saved"]
        log_box.code("\n".join(logs), language=None)
        status.empty()

        # Force cache refresh
        st.session_state.model_version += 1
        load_recognizer.clear()

        st.markdown(f"""
        <div style="background:#0d2b1a;border:2px solid #00c853;border-radius:12px;
             padding:1.2rem;color:#00e676;font-size:1rem;font-weight:700;
             text-align:center;margin-top:1rem;">
            ✅ Training Successful!<br>
            <span style="font-size:0.84rem;font-weight:400;color:#69f0ae;">
                {len(faces_list)} images · {len(label_map)} persons trained
            </span>
        </div>""", unsafe_allow_html=True)
        st.balloons()

    except Exception as e:
        st.error(f"❌ Training error: {e}")


# ══════════════════════════════════════════════════════════════
#  TAB 3 — FACE RECOGNITION
# ══════════════════════════════════════════════════════════════
def tab_recognition():
    st.markdown("""
    <div class="card">
        <div class="card-title">👁️ Step 3 — Live Face Recognition</div>
        <span style="color:#888aaa;font-size:0.87rem;">
            Take a photo — the system will detect and identify the face.
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Model check ───────────────────────────────────────────
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        st.markdown("""
        <div class="badge-warn" style="padding:1rem;">
            ❌ Model not found! Please train the model in <b>Tab 2</b> first.
        </div>""", unsafe_allow_html=True)
        return

    recognizer, label_map = load_recognizer(st.session_state.model_version)
    if recognizer is None:
        if label_map.get("__error__") == "contrib":
            st.error("""
❌ **opencv-contrib-python not found!**

Run this in terminal:

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

Then restart Streamlit.
""")
        else:
            st.error("❌ Model could not be loaded. Please train the model in Tab 2 first.")
        return

    persons_str = "  ·  ".join(label_map.values())
    st.markdown(f"""
    <div class="badge-ok" style="margin-bottom:1rem;">
        ✅ Model Ready &nbsp;|&nbsp; Persons: <b>{persons_str}</b>
    </div>""", unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────
    threshold = CONFIDENCE_THRESHOLD
    show_conf = True
    with st.expander("⚙️ Recognition Settings"):
        threshold = st.slider(
            "Confidence Threshold", 40, 130, CONFIDENCE_THRESHOLD, 5,
            help="Lower = stricter. Lower value = stricter match. Too many unknowns → increase it."
        )
        show_conf = st.checkbox("Confidence score dikhao", value=True)

    st.markdown("---")

    # ── Layout: Camera | Results ──────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("#### 📷 Camera")
        cam_img = st.camera_input("Photo lo", key="recog_cam",
                                  label_visibility="collapsed")

    with right:
        st.markdown("#### 🔍 Results")
        result_ph = st.empty()
        if not cam_img:
            result_ph.markdown("""
            <div class="result-noface">
                👈 Camera se photo lo
            </div>""", unsafe_allow_html=True)

    # ── Process image ─────────────────────────────────────────
    if cam_img:
        img_bgr = camera_bytes_to_bgr(cam_img)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        gray_bl = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            gray_bl, scaleFactor=1.05, minNeighbors=5,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        annotated = img_bgr.copy()
        results   = []

        for (x, y, w, h) in faces:
            face_gray    = gray[y:y+h, x:x+w]
            face_eq      = cv2.equalizeHist(face_gray)
            face_resized = cv2.resize(face_eq, FACE_SIZE,
                                      interpolation=cv2.INTER_CUBIC)

            label, conf = recognizer.predict(face_resized)

            if conf < threshold:
                name  = label_map.get(label, "Unknown")
                color = (0, 210, 80)
                known = True
            else:
                name  = "Unknown"
                color = (30, 30, 210)
                known = False

            # Annotate frame
            txt = f"{name}  ({conf:.1f})" if show_conf else name
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(annotated, (x, y-th-14), (x+tw+6, y), color, -1)
            cv2.putText(annotated, txt, (x+3, y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Confidence bar below face
            bar_max  = 120
            bar_fill = max(0, int((1 - conf / bar_max) * w))
            cv2.rectangle(annotated, (x, y+h+3), (x+w, y+h+11), (30,30,30), -1)
            cv2.rectangle(annotated, (x, y+h+3), (x+bar_fill, y+h+11), color, -1)

            results.append({
                "name": name, "conf": conf, "known": known,
                "crop": img_bgr[y:y+h, x:x+w]
            })

        with left:
            st.image(bgr_to_pil(annotated), width='stretch')

        with right:
            result_ph.empty()
            if not results:
                st.markdown("""
                <div class="result-noface">
                    ❌ No face detected!<br>
                    <span style="font-size:0.8rem;">
                        Move closer and ensure good lighting
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                for i, r in enumerate(results):
                    if r["known"]:
                        st.markdown(f"""
                        <div class="result-known">
                            ✅ {r['name']}<br>
                            <span style="font-size:0.8rem;font-weight:400;color:#69f0ae;">
                                Confidence: {r['conf']:.1f}
                                {'&nbsp;(lower = better)' if show_conf else ''}
                            </span>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-unknown">
                            ❓ Person not recognized<br>
                            <span style="font-size:0.8rem;font-weight:400;color:#ff8a80;">
                                Score: {r['conf']:.1f} &nbsp;|&nbsp; Threshold: {threshold}
                            </span>
                        </div>""", unsafe_allow_html=True)

                    if r["crop"] is not None and r["crop"].size > 0:
                        st.image(bgr_to_pil(cv2.resize(r["crop"], (90, 90))),
                                 caption=f"Face {i+1}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    # Header
    st.markdown("""
    <div class="header-banner">
        <div style="font-size:2.4rem;">🎯</div>
        <div>
            <h1>Face Recognition System</h1>
            <p>Create Dataset → Train Model → Recognize Faces</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # Step indicator
    st.markdown("""
    <div class="steps">
        <div class="step">📸<b>Dataset Create</b></div>
        <div class="arrow">→</div>
        <div class="step">🧠<b>Train Model</b></div>
        <div class="arrow">→</div>
        <div class="step">👁️<b>Face Recognition</b></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📸  Dataset Create",
        "🧠  Train Model",
        "👁️  Face Recognition",
    ])
    with tab1: tab_dataset()
    with tab2: tab_train()
    with tab3: tab_recognition()


if __name__ == "__main__":
    main()
