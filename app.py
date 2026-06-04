import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Image Classifier",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette per category (cycles if more than defined) ───────────────────
CATEGORY_COLORS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#ec4899",  # pink
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#06b6d4",  # cyan
]

# Bar fill colors for top-3 ranks
BAR_COLORS = ["#6366f1", "#10b981", "#f59e0b"]

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fdf4ff 50%, #f0fdf9 100%);
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Page header ── */
.page-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 2px solid #e2e5ea;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}
.page-header p {
    font-size: 0.9rem;
    color: #6b7280;
    margin: 0.3rem 0 0 0;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.85);
    border: 1px solid #e2e5ea;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(8px);
}

/* ── Prediction badge ── */
.pred-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}

/* ── Confidence bars ── */
.conf-row { margin-bottom: 0.75rem; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    color: #374151;
    margin-bottom: 0.3rem;
    font-weight: 500;
}
.conf-track {
    background: #ede9fe;
    border-radius: 6px;
    height: 9px;
    width: 100%;
}
.conf-fill {
    height: 9px;
    border-radius: 6px;
    transition: width 0.7s ease;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
}

/* ── History item ── */
.hist-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0;
    border-bottom: 1px solid #f0f2f5;
    font-size: 0.85rem;
    color: #374151;
}
.hist-item:last-child { border-bottom: none; }
.hist-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    background: linear-gradient(135deg, #ede9fe, #fce7f3);
    color: #6366f1;
    padding: 0.22rem 0.6rem;
    border-radius: 5px;
    font-weight: 600;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%);
    border-right: 1px solid #e2e5ea;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* ── Category pills ── */
.cat-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.84rem;
    font-weight: 500;
    border: 1px solid transparent;
}
.cat-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Divider ── */
.thin-divider {
    border: none;
    border-top: 1px solid #e2e5ea;
    margin: 1.25rem 0;
}

/* ── About block ── */
.about-block { font-size: 0.85rem; color: #4b5563; line-height: 1.7; }
.about-block strong { color: #6366f1; }

/* ── Metric row ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-box {
    flex: 1;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    border: 1px solid transparent;
}
.metric-box .val {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}
.metric-box .lbl {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.1rem;
}

/* Metric box colour variants */
.mb-indigo  { background: #ede9fe; border-color: #c4b5fd; }
.mb-indigo  .val { color: #4f46e5; }
.mb-indigo  .lbl { color: #7c3aed; }

.mb-emerald { background: #d1fae5; border-color: #6ee7b7; }
.mb-emerald .val { color: #059669; }
.mb-emerald .lbl { color: #047857; }

.mb-amber   { background: #fef3c7; border-color: #fcd34d; }
.mb-amber   .val { color: #d97706; }
.mb-amber   .lbl { color: #b45309; }

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: #9ca3af;
    background: rgba(255,255,255,0.7);
    border: 2px dashed #c4b5fd;
    border-radius: 12px;
}
.empty-state .icon { font-size: 2.8rem; margin-bottom: 0.75rem; }
.empty-state .msg  { font-size: 0.9rem; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeUp 0.45s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('ecommerce_classifier.keras')
    with open('class_names.json') as f:
        class_names = json.load(f)
    return model, class_names

with st.spinner("Loading model…"):
    model, CLASS_NAMES = load_model()

NUM_CLASSES = len(CLASS_NAMES)

# Build per-category color map (cycles if > 10 categories)
CAT_COLOR_MAP = {
    cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
    for i, cat in enumerate(CLASS_NAMES)
}

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏷️ Product Classifier</div>', unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Supported Categories</div>', unsafe_allow_html=True)
    for cat in CLASS_NAMES:
        color = CAT_COLOR_MAP[cat]
        label = cat.replace('_', ' ').title()
        # Light tint background = color + "18" (hex opacity ~10%)
        bg    = color + "15"
        st.markdown(f"""
        <div class="cat-pill" style="background:{bg}; border-color:{color}40; color:#374151;">
            <div class="cat-dot" style="background:{color};"></div>
            {label}
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">About the Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="about-block">
        <strong>Architecture:</strong> MobileNetV2 (transfer learning)<br><br>
        <strong>Input size:</strong> 224 × 224 px<br><br>
        <strong>Classes:</strong> {NUM_CLASSES} product categories<br><br>
        <strong>Training:</strong> ImageNet pre-trained weights fine-tuned on the
        <em>E-Commerce Product Images 18K</em> dataset with augmentation and
        class-weight balancing.<br><br>
        <strong>Framework:</strong> TensorFlow / Keras
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    if st.button("🗑  Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header fade-in">
    <h1>E-Commerce Product Image Classifier</h1>
    <p>Upload a product photo and the model will identify its category instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── Metric row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row fade-in">
    <div class="metric-box mb-indigo">
        <div class="val">{NUM_CLASSES}</div>
        <div class="lbl">Categories</div>
    </div>
    <div class="metric-box mb-emerald">
        <div class="val">224px</div>
        <div class="lbl">Input Size</div>
    </div>
    <div class="metric-box mb-amber">
        <div class="val">{len(st.session_state.history)}</div>
        <div class="lbl">Predictions Made</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True, caption=uploaded.name)

with right_col:
    st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

    if uploaded:
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        with st.spinner("Analysing image…"):
            time.sleep(0.3)
            preds = model.predict(arr, verbose=0)[0]

        top3_idx = np.argsort(preds)[::-1][:3]
        top_idx  = top3_idx[0]
        top_conf = preds[top_idx]
        top_name = CLASS_NAMES[top_idx].replace('_', ' ').title()
        top_color = CAT_COLOR_MAP[CLASS_NAMES[top_idx]]

        # Avoid duplicate history entries for the same upload
        if not st.session_state.history or st.session_state.history[0][0] != uploaded.name:
            st.session_state.history.insert(0, (uploaded.name, top_name, float(top_conf)))
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]

        # Prediction badge — uses the category's own color
        st.markdown(f"""
        <div class="fade-in">
            <div class="pred-badge" style="background: linear-gradient(135deg, {top_color}, {top_color}cc);
                 box-shadow: 0 4px 14px {top_color}44;">
                ✅ &nbsp;{top_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-3 confidence bars with distinct colors
        medals = ["🥇", "🥈", "🥉"]
        for rank, idx in enumerate(top3_idx):
            name   = CLASS_NAMES[idx].replace('_', ' ').title()
            score  = preds[idx] * 100
            bcolor = BAR_COLORS[rank]
            track_bg = bcolor + "22"
            st.markdown(f"""
            <div class="conf-row fade-in">
                <div class="conf-label">
                    <span>{medals[rank]} {name}</span>
                    <span style="color:{bcolor}; font-family:'DM Mono',monospace;">{score:.1f}%</span>
                </div>
                <div class="conf-track" style="background:{track_bg};">
                    <div class="conf-fill" style="width:{score:.1f}%; background:{bcolor};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state fade-in">
            <div class="icon">📦</div>
            <div class="msg">Upload an image on the left<br>to see predictions here.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Upload history ─────────────────────────────────────────────────────────────
st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Recent Predictions</div>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for fname, pred, conf in st.session_state.history:
        # Find matching color for this prediction
        raw_key = next((k for k in CAT_COLOR_MAP if k.replace('_',' ').title() == pred), None)
        hcolor  = CAT_COLOR_MAP.get(raw_key, "#6366f1") if raw_key else "#6366f1"
        hbg     = hcolor + "15"
        st.markdown(f"""
        <div class="hist-item">
            <span style="color:#6b7280; font-size:0.82rem; max-width:50%;
                         overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                📄 {fname}
            </span>
            <span class="hist-tag" style="background:{hbg}; color:{hcolor};">{pred}</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.8rem;
                         color:{hcolor}; font-weight:600;">
                {conf*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="font-size:0.85rem; color:#9ca3af; padding:0.5rem 0;">
        No predictions yet — upload an image above to get started.
    </div>
    """, unsafe_allow_html=True)