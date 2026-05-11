import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import io
import base64
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="DigitAI - Handwritten Digit Recognition",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
  --bg-base:        #F7F8FA;
  --bg-surface:     #FFFFFF;
  --bg-subtle:      #F0F2F5;
  --border:         #E4E8EF;
  --text-primary:   #0F1523;
  --text-secondary: #5A6478;
  --text-muted:     #9CA3AF;
  --accent:         #3B6EF8;
  --accent-light:   #EEF2FF;
  --accent-hover:   #2D5CE8;
  --shadow-sm:      0 1px 3px rgba(15,21,35,.06), 0 1px 2px rgba(15,21,35,.04);
  --radius-sm:      8px;
  --radius-md:      12px;
}

* { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Poppins', sans-serif !important;
  color: var(--text-primary);
}
.stApp { background: var(--bg-base); }
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem; }
[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
  font-family: 'Poppins', sans-serif !important;
}

[data-testid="stMetric"] {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px 20px !important;
  box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] {
  font-size: 11px !important; font-weight: 600 !important;
  color: var(--text-muted) !important; letter-spacing: 0.06em; text-transform: uppercase;
}
[data-testid="stMetricValue"] {
  color: var(--text-primary) !important;
  font-size: 26px !important; font-weight: 700 !important;
}

.stButton > button {
  background: var(--accent) !important;
  border: none !important; color: #fff !important;
  font-family: 'Poppins', sans-serif !important;
  font-size: 13px !important; font-weight: 600 !important;
  border-radius: var(--radius-sm) !important;
  width: 100%; padding: 10px 20px !important;
  box-shadow: 0 2px 8px rgba(59,110,248,.25) !important;
  transition: background 0.2s ease !important;
}
.stButton > button:hover {
  background: var(--accent-hover) !important;
  box-shadow: 0 4px 16px rgba(59,110,248,.35) !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-subtle); border-radius: var(--radius-md);
  border: 1px solid var(--border); padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Poppins', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--text-secondary) !important;
  border-radius: var(--radius-sm) !important; padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-surface) !important;
  color: var(--accent) !important;
  box-shadow: var(--shadow-sm) !important; font-weight: 600 !important;
}

.stProgress > div > div { background: var(--accent) !important; border-radius: 99px !important; }
.stProgress > div {
  background: var(--bg-subtle) !important; border-radius: 99px !important;
  border: 1px solid var(--border); height: 8px !important;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()


def preprocess_upload(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    arr = np.array(image)
    if arr.mean() > 127:
        image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image = ImageOps.expand(image, border=20, fill=0)
    image = image.resize((28, 28), Image.LANCZOS)
    arr = np.array(image) / 255.0
    return arr.reshape(1, 28, 28, 1)


def preprocess_canvas(image_data: np.ndarray) -> np.ndarray:
    img = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.expand(img, border=20, fill=0)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img) / 255.0
    return arr.reshape(1, 28, 28, 1)


def predict_from_array(arr: np.ndarray):
    t0 = time.time()
    probs = model.predict(arr, verbose=0)[0]
    ms = round((time.time() - t0) * 1000, 1)
    digit = int(np.argmax(probs))
    conf = round(float(np.max(probs)) * 100, 1)
    top3 = [(int(i), round(float(probs[i]) * 100, 1)) for i in np.argsort(probs)[::-1][:3]]
    return digit, conf, probs.tolist(), ms, top3


if "history" not in st.session_state:
    st.session_state.history = []
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


st.markdown("""
<div style="padding:16px 0 24px;">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
    <div style="width:40px;height:40px;border-radius:10px;
      background:linear-gradient(135deg,#3B6EF8,#6B8FFF);
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 4px 12px rgba(59,110,248,0.3);font-size:20px;flex-shrink:0;color:#fff;font-weight:700;">D</div>
    <div>
      <h1 style="font-family:'Poppins',sans-serif;font-size:26px;font-weight:700;
        color:#0F1523;margin:0;line-height:1.2;letter-spacing:-0.02em;">
        DigitAI
        <span style="font-size:13px;font-weight:500;color:#3B6EF8;
          background:#EEF2FF;border-radius:6px;padding:3px 10px;
          margin-left:10px;vertical-align:middle;">CNN · MNIST</span>
      </h1>
      <p style="font-family:'Poppins',sans-serif;font-size:13px;color:#5A6478;margin:0;">
        Handwritten digit recognition — 99.2% test accuracy
      </p>
    </div>
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#E4E8EF 0%,transparent 100%);margin-bottom:28px;"></div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:32px;height:32px;border-radius:8px;
          background:linear-gradient(135deg,#3B6EF8,#6B8FFF);
          display:flex;align-items:center;justify-content:center;
          font-size:14px;color:white;font-weight:700;flex-shrink:0;">D</div>
        <span style="font-family:'Poppins',sans-serif;font-size:15px;font-weight:700;color:#0F1523;">DigitAI</span>
      </div>
      <div style="background:#FFFFFF;border:1px solid #E4E8EF;border-radius:12px;
        padding:14px 16px;margin-bottom:6px;box-shadow:0 1px 3px rgba(15,21,35,.06);">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
          <div style="width:8px;height:8px;border-radius:50%;
            background:#12B76A;box-shadow:0 0 0 2px #D1FAE5;flex-shrink:0;"></div>
          <span style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;color:#065F46;">
            Model Active
          </span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
          font-family:'Poppins',sans-serif;font-size:11px;color:#5A6478;line-height:1.6;">
          <div><span style="color:#9CA3AF;font-size:10px;font-weight:600;text-transform:uppercase;">Architecture</span><br/>CNN (Keras)</div>
          <div><span style="color:#9CA3AF;font-size:10px;font-weight:600;text-transform:uppercase;">Dataset</span><br/>MNIST</div>
          <div><span style="color:#9CA3AF;font-size:10px;font-weight:600;text-transform:uppercase;">Input</span><br/>28 x 28 px</div>
          <div><span style="color:#9CA3AF;font-size:10px;font-weight:600;text-transform:uppercase;">Classes</span><br/>0 - 9</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Poppins',sans-serif;font-size:11px;font-weight:600;
      color:#9CA3AF;text-transform:uppercase;letter-spacing:.08em;margin:8px 0 12px;">
      Prediction History</div>""", unsafe_allow_html=True)

    if st.session_state.history:
        for h in reversed(st.session_state.history[-8:]):
            badge_bg    = "#ECFDF5" if h["conf"] >= 90 else "#FFFBEB" if h["conf"] >= 70 else "#FEF2F2"
            badge_color = "#065F46" if h["conf"] >= 90 else "#92400E" if h["conf"] >= 70 else "#991B1B"
            bar_color   = "#12B76A" if h["conf"] >= 90 else "#F59E0B" if h["conf"] >= 70 else "#EF4444"
            st.markdown(f"""
            <div style="background:#FFF;border:1px solid #E4E8EF;border-radius:10px;
              padding:10px 12px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
                <span style="font-family:'Poppins',sans-serif;font-size:28px;font-weight:700;color:#0F1523;line-height:1;">
                  {h["digit"]}</span>
                <span style="font-family:'Poppins',sans-serif;font-size:11px;font-weight:600;
                  background:{badge_bg};color:{badge_color};border-radius:6px;padding:2px 8px;">
                  {h["conf"]}%</span>
              </div>
              <div style="background:#F0F2F5;border-radius:99px;height:3px;overflow:hidden;">
                <div style="width:{int(h["conf"])}%;height:100%;background:{bar_color};border-radius:99px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="background:#F7F8FA;border:1px dashed #E4E8EF;border-radius:10px;
          padding:28px 16px;text-align:center;margin-top:4px;">
          <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:500;color:#9CA3AF;">
            No predictions yet</div>
          <div style="font-family:'Poppins',sans-serif;font-size:11px;color:#C0C6D0;margin-top:4px;">
            Upload or draw a digit to get started</div>
        </div>""", unsafe_allow_html=True)


col_input, col_output = st.columns([1, 1], gap="large")

result = None

with col_input:
    st.markdown("""<div style="font-family:'Poppins',sans-serif;font-size:14px;font-weight:600;
      color:#0F1523;margin-bottom:16px;display:flex;align-items:center;gap:8px;">
      <div style="width:6px;height:20px;background:#3B6EF8;border-radius:3px;flex-shrink:0;"></div>
      Input</div>""", unsafe_allow_html=True)

    tab_upload, tab_draw = st.tabs(["Upload Image", "Draw Digit"])

    with tab_upload:
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["png", "jpg", "jpeg", "bmp", "gif"],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read()))
            st.image(img, caption="Uploaded image", use_column_width=True)
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            if st.button("Analyze Image", key="analyze_upload"):
                with st.spinner("Running inference..."):
                    arr = preprocess_upload(img)
                    result = predict_from_array(arr)
        else:
            st.markdown("""
            <div style="background:#F7F8FA;border:2px dashed #E4E8EF;border-radius:12px;
              padding:36px 20px;text-align:center;margin-top:12px;">
              <div style="font-family:'Poppins',sans-serif;font-size:13px;font-weight:500;
                color:#5A6478;margin-bottom:4px;">Drop your image here</div>
              <div style="font-family:'Poppins',sans-serif;font-size:11px;color:#9CA3AF;">
                Supports PNG, JPG, BMP, GIF</div>
            </div>""", unsafe_allow_html=True)

    with tab_draw:
        st.markdown("""<div style="font-family:'Poppins',sans-serif;font-size:12px;
          color:#5A6478;margin:12px 0 10px;line-height:1.5;">
          Draw a digit (0-9) on the canvas. Use Undo/Redo to fix mistakes.</div>""",
          unsafe_allow_html=True)

        stroke_width = st.slider("Brush size", 10, 30, 20, key="stroke_width")

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Undo", key="undo_btn"):
                if st.session_state.canvas_key > 0:
                    st.session_state.canvas_key -= 1
                    st.rerun()
        with btn_col2:
            if st.button("Clear", key="clear_btn"):
                st.session_state.canvas_key += 1
                st.rerun()
        with btn_col3:
            predict_clicked = st.button("Predict", key="predict_drawing")

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=False,
        )

        if predict_clicked:
            if canvas_result.image_data is not None:
                pixel_sum = canvas_result.image_data[:, :, :3].sum()
                if pixel_sum == 0:
                    st.warning("Canvas is empty. Please draw a digit first.")
                else:
                    with st.spinner("Running inference..."):
                        arr = preprocess_canvas(canvas_result.image_data)
                        result = predict_from_array(arr)
            else:
                st.warning("Please draw a digit first.")


with col_output:
    st.markdown("""<div style="font-family:'Poppins',sans-serif;font-size:14px;font-weight:600;
      color:#0F1523;margin-bottom:16px;display:flex;align-items:center;gap:8px;">
      <div style="width:6px;height:20px;background:#3B6EF8;border-radius:3px;flex-shrink:0;"></div>
      Prediction</div>""", unsafe_allow_html=True)

    if result:
        digit, conf, probs, ms, top3 = result
        st.session_state.history.append({"digit": digit, "conf": conf})

        if conf < 60:
            st.markdown("""<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;
              padding:10px 14px;margin-bottom:14px;">
              <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;color:#991B1B;">
                Low Confidence</div>
              <div style="font-family:'Poppins',sans-serif;font-size:11px;color:#B91C1C;">
                Try redrawing the digit larger and centered.</div></div>""",
                unsafe_allow_html=True)
        elif conf < 80:
            st.markdown("""<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;
              padding:10px 14px;margin-bottom:14px;">
              <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;color:#92400E;">
                Moderate Confidence</div>
              <div style="font-family:'Poppins',sans-serif;font-size:11px;color:#B45309;">
                Drawing more clearly may improve accuracy.</div></div>""",
                unsafe_allow_html=True)

        result_color  = "#12B76A" if conf >= 90 else "#F59E0B" if conf >= 70 else "#EF4444"
        result_bg     = "#ECFDF5" if conf >= 90 else "#FFFBEB" if conf >= 70 else "#FEF2F2"
        result_border = "#A7F3D0" if conf >= 90 else "#FDE68A" if conf >= 70 else "#FECACA"

        st.markdown(f"""
        <div style="background:#fff;border:1px solid #E4E8EF;border-radius:16px;
          padding:28px 24px;text-align:center;margin-bottom:16px;
          box-shadow:0 4px 16px rgba(15,21,35,.06);">
          <div style="display:inline-flex;align-items:center;gap:6px;
            background:{result_bg};border:1px solid {result_border};
            border-radius:99px;padding:4px 12px;margin-bottom:16px;">
            <div style="width:6px;height:6px;border-radius:50%;background:{result_color};"></div>
            <span style="font-family:'Poppins',sans-serif;font-size:11px;font-weight:600;
              color:{result_color};">{conf}% Confidence</span>
          </div>
          <div style="font-family:'Poppins',sans-serif;font-size:96px;font-weight:700;
            line-height:1;color:#0F1523;letter-spacing:-.04em;margin-bottom:12px;">
            {digit}</div>
          <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:500;color:#9CA3AF;">
            Recognized Digit</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Digit", str(digit))
        with m2: st.metric("Confidence", f"{conf}%")
        with m3: st.metric("Inference", f"{ms} ms")

        st.markdown("""<div style="font-family:'Poppins',sans-serif;font-size:11px;font-weight:600;
          color:#9CA3AF;text-transform:uppercase;letter-spacing:.06em;margin:16px 0 6px;">
          Confidence Score</div>""", unsafe_allow_html=True)
        st.progress(int(conf))

        st.markdown("""<div style="font-family:'Poppins',sans-serif;font-size:11px;font-weight:600;
          color:#9CA3AF;text-transform:uppercase;letter-spacing:.06em;margin:20px 0 10px;">
          Top 3 Predictions</div>""", unsafe_allow_html=True)

        t1, t2, t3 = st.columns(3)
        rank_styles = [
            ("#EEF2FF", "#3B6EF8", "#3B6EF8"),
            ("#F7F8FA", "#5A6478", "#9CA3AF"),
            ("#F7F8FA", "#5A6478", "#9CA3AF"),
        ]
        for col, (d, c), (bg, nc, pc) in zip([t1, t2, t3], top3, rank_styles):
            with col:
                st.markdown(f"""
                <div style="background:{bg};border:1px solid #E4E8EF;border-radius:10px;
                  padding:12px 8px;text-align:center;">
                  <div style="font-family:'Poppins',sans-serif;font-size:30px;font-weight:700;
                    color:{nc};line-height:1;">{d}</div>
                  <div style="font-family:'Poppins',sans-serif;font-size:11px;font-weight:600;
                    color:{pc};margin-top:4px;">{c}%</div>
                </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E4E8EF;border-radius:16px;
          padding:60px 32px;text-align:center;box-shadow:0 1px 4px rgba(15,21,35,.04);">
          <div style="font-family:'Poppins',sans-serif;font-size:14px;font-weight:600;
            color:#5A6478;margin-bottom:6px;">Awaiting Input</div>
          <div style="font-family:'Poppins',sans-serif;font-size:12px;color:#9CA3AF;
            max-width:220px;margin:0 auto;line-height:1.6;">
            Upload an image or draw a digit on the left to see the prediction here.</div>
        </div>""", unsafe_allow_html=True)


st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""<div style="display:flex;align-items:center;gap:8px;font-family:'Poppins',sans-serif;
  font-size:14px;font-weight:600;color:#0F1523;margin-bottom:16px;">
  <div style="width:6px;height:20px;background:#3B6EF8;border-radius:3px;flex-shrink:0;"></div>
  Class Probability Distribution</div>""", unsafe_allow_html=True)

if result:
    digit, conf, probs, ms, top3 = result
    cols = st.columns(10)
    for i, col in enumerate(cols):
        pct    = round(probs[i] * 100, 1)
        is_top = (i == digit)
        bg     = "#EEF2FF" if is_top else "#FFFFFF"
        border = "#C7D7FD" if is_top else "#E4E8EF"
        num_c  = "#3B6EF8" if is_top else "#0F1523"
        pct_c  = "#3B6EF8" if is_top else "#9CA3AF"
        bar_c  = "#3B6EF8" if is_top else "#CBD5E1"
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:10px;
              padding:10px 4px;text-align:center;">
              <div style="font-family:'Poppins',sans-serif;font-size:20px;font-weight:700;
                color:{num_c};line-height:1;">{i}</div>
              <div style="font-family:'Poppins',sans-serif;font-size:9px;font-weight:600;
                color:{pct_c};margin-top:4px;">{pct}%</div>
              <div style="background:#F0F2F5;border-radius:99px;height:4px;margin-top:8px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{bar_c};border-radius:99px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E4E8EF;border-radius:12px;
      padding:28px;text-align:center;">
      <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:500;color:#9CA3AF;">
        Run an analysis to see the full probability distribution.</div>
    </div>""", unsafe_allow_html=True)


st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""<div style="display:flex;align-items:center;gap:8px;font-family:'Poppins',sans-serif;
  font-size:14px;font-weight:600;color:#0F1523;margin-bottom:16px;">
  <div style="width:6px;height:20px;background:#3B6EF8;border-radius:3px;flex-shrink:0;"></div>
  Model Specifications</div>""", unsafe_allow_html=True)

specs = [
    ("CNN",    "Architecture"),
    ("10",     "Epochs"),
    ("60,000", "Train Images"),
    ("10,000", "Test Images"),
    ("99.2%",  "Accuracy"),
    ("28x28",  "Input Shape"),
    ("Adam",   "Optimizer"),
    ("10",     "Output Classes"),
]
scols = st.columns(8)
for col, (val, key) in zip(scols, specs):
    with col:
        st.metric(key, val)
