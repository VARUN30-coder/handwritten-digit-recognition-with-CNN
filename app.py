import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import io
import base64
from streamlit.components.v1 import html as st_html

st.set_page_config(
    page_title="DigitAI — Handwritten Digit Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* ── Design Tokens ── */
:root {
  --bg-base:      #F7F8FA;
  --bg-surface:   #FFFFFF;
  --bg-subtle:    #F0F2F5;
  --border:       #E4E8EF;
  --border-focus: #3B6EF8;
  --text-primary: #0F1523;
  --text-secondary: #5A6478;
  --text-muted:   #9CA3AF;
  --accent:       #3B6EF8;
  --accent-light: #EEF2FF;
  --accent-hover: #2D5CE8;
  --success:      #12B76A;
  --success-light:#ECFDF5;
  --warning:      #F59E0B;
  --warning-light:#FFFBEB;
  --danger:       #EF4444;
  --danger-light: #FEF2F2;
  --shadow-sm:    0 1px 3px rgba(15,21,35,.06), 0 1px 2px rgba(15,21,35,.04);
  --shadow-md:    0 4px 12px rgba(15,21,35,.08), 0 2px 4px rgba(15,21,35,.04);
  --shadow-lg:    0 12px 32px rgba(15,21,35,.10), 0 4px 8px rgba(15,21,35,.04);
  --radius-sm:    8px;
  --radius-md:    12px;
  --radius-lg:    16px;
  --radius-xl:    20px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Poppins', sans-serif !important;
  color: var(--text-primary);
}

.stApp {
  background: var(--bg-base);
}

.main .block-container {
  padding: 2rem 2.5rem 3rem;
  max-width: 1280px;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
  padding: 0;
}
[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1.25rem;
}
[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
  font-family: 'Poppins', sans-serif !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px 20px !important;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
  box-shadow: var(--shadow-md);
}
[data-testid="stMetricLabel"] {
  font-family: 'Poppins', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
[data-testid="stMetricValue"] {
  font-family: 'Poppins', sans-serif !important;
  color: var(--text-primary) !important;
  font-size: 26px !important;
  font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  border: none !important;
  color: #fff !important;
  font-family: 'Poppins', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  border-radius: var(--radius-sm) !important;
  width: 100%;
  padding: 10px 20px !important;
  transition: background 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(59,110,248,.25) !important;
}
.stButton > button:hover {
  background: var(--accent-hover) !important;
  box-shadow: 0 4px 16px rgba(59,110,248,.35) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}

/* Clear/secondary button override */
.btn-secondary .stButton > button {
  background: var(--bg-subtle) !important;
  color: var(--text-secondary) !important;
  box-shadow: none !important;
}
.btn-secondary .stButton > button:hover {
  background: var(--border) !important;
  box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-subtle);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  padding: 4px;
  gap: 2px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Poppins', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 20px !important;
  transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-surface) !important;
  color: var(--accent) !important;
  border-bottom: none !important;
  box-shadow: var(--shadow-sm) !important;
  font-weight: 600 !important;
}

/* ── Progress Bar ── */
.stProgress > div > div {
  background: var(--accent) !important;
  border-radius: 99px !important;
  transition: width 0.4s ease !important;
}
.stProgress > div {
  background: var(--bg-subtle) !important;
  border-radius: 99px !important;
  border: 1px solid var(--border);
  height: 8px !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg-surface);
  border: 2px dashed var(--border);
  border-radius: var(--radius-md);
  padding: 24px;
  transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent);
  background: var(--accent-light);
}

/* ── Alerts ── */
.stSuccess {
  background: var(--success-light) !important;
  border: 1px solid #A7F3D0 !important;
  border-radius: var(--radius-md) !important;
  color: #065F46 !important;
}
.stInfo {
  background: var(--accent-light) !important;
  border: 1px solid #C7D7FD !important;
  border-radius: var(--radius-md) !important;
}
.stWarning {
  background: var(--warning-light) !important;
  border: 1px solid #FDE68A !important;
  border-radius: var(--radius-md) !important;
}
.stError {
  background: var(--danger-light) !important;
  border: 1px solid #FECACA !important;
  border-radius: var(--radius-md) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Poppins', sans-serif !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
  border-top-color: var(--accent) !important;
}

/* ── Text Input ── */
[data-testid="stTextInput"] input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-secondary) !important;
  font-family: 'Poppins', sans-serif !important;
  font-size: 12px !important;
}

/* ── Image ── */
[data-testid="stImage"] {
  border-radius: var(--radius-md);
  overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

def preprocess_upload(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.LANCZOS)
    arr = np.array(image) / 255.0
    return arr.reshape(1, 28, 28, 1)

def preprocess_canvas(canvas_arr: np.ndarray) -> np.ndarray:
    gray = canvas_arr[:, :, 0].astype(np.uint8)
    image = Image.fromarray(gray, mode="L")
    image = image.resize((28, 28), Image.LANCZOS)
    arr = np.array(image) / 255.0
    return arr.reshape(1, 28, 28, 1)

def predict_from_array(arr: np.ndarray):
    t0 = time.time()
    probs = model.predict(arr, verbose=0)[0]
    ms = round((time.time() - t0) * 1000, 1)
    digit = int(np.argmax(probs))
    conf  = round(float(np.max(probs)) * 100, 1)
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(int(i), round(float(probs[i]) * 100, 1)) for i in top3_idx]
    return digit, conf, probs.tolist(), ms, top3

if "history" not in st.session_state:
    st.session_state.history = []
if "preview" not in st.session_state:
    st.session_state.preview = None

st.markdown("""
<div style="padding: 16px 0 24px;">
  <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
    <div style="
      width:40px; height:40px; border-radius:10px;
      background: linear-gradient(135deg, #3B6EF8, #6B8FFF);
      display:flex; align-items:center; justify-content:center;
      box-shadow: 0 4px 12px rgba(59,110,248,0.3);
      font-size:18px; flex-shrink:0;">
      ✦
    </div>
    <div>
      <h1 style="
        font-family:'Poppins',sans-serif; font-size:26px; font-weight:700;
        color:#0F1523; margin:0; line-height:1.2; letter-spacing:-0.02em;">
        DigitAI
        <span style="
          font-size:13px; font-weight:500; color:#3B6EF8;
          background:#EEF2FF; border-radius:6px; padding:3px 10px;
          margin-left:10px; vertical-align:middle; letter-spacing:0;">
          CNN · MNIST
        </span>
      </h1>
      <p style="
        font-family:'Poppins',sans-serif; font-size:13px; font-weight:400;
        color:#5A6478; margin:0;">
        Handwritten digit recognition — 99.2% test accuracy
      </p>
    </div>
  </div>
</div>
<div style="height:1px; background:linear-gradient(90deg,#E4E8EF 0%,transparent 100%); margin-bottom:28px;"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
        <div style="
          width:32px; height:32px; border-radius:8px;
          background:linear-gradient(135deg,#3B6EF8,#6B8FFF);
          display:flex; align-items:center; justify-content:center;
          font-size:14px; color:white; flex-shrink:0;">
          ✦
        </div>
        <span style="font-family:'Poppins',sans-serif; font-size:15px; font-weight:700; color:#0F1523;">DigitAI</span>
      </div>

      <div style="
        background:#FFFFFF; border:1px solid #E4E8EF; border-radius:12px;
        padding:14px 16px; margin-bottom:6px;
        box-shadow:0 1px 3px rgba(15,21,35,.06);">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
          <div style="
            width:8px; height:8px; border-radius:50%;
            background:#12B76A; box-shadow:0 0 0 2px #D1FAE5; flex-shrink:0;"></div>
          <span style="font-family:'Poppins',sans-serif; font-size:12px; font-weight:600; color:#065F46;">
            Model Active
          </span>
        </div>
        <div style="
          display:grid; grid-template-columns:1fr 1fr; gap:8px;
          font-family:'Poppins',sans-serif; font-size:11px; color:#5A6478; line-height:1.6;">
          <div><span style="color:#9CA3AF; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em;">Architecture</span><br/>CNN (Keras)</div>
          <div><span style="color:#9CA3AF; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em;">Dataset</span><br/>MNIST</div>
          <div><span style="color:#9CA3AF; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em;">Input</span><br/>28 × 28 px</div>
          <div><span style="color:#9CA3AF; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em;">Classes</span><br/>0 – 9</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
      font-family:'Poppins',sans-serif; font-size:11px; font-weight:600;
      color:#9CA3AF; text-transform:uppercase; letter-spacing:.08em;
      margin: 8px 0 12px;">
      Prediction History
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        for h in reversed(st.session_state.history[-8:]):
            if h['conf'] >= 90:
                badge_bg, badge_color = "#ECFDF5", "#065F46"
            elif h['conf'] >= 70:
                badge_bg, badge_color = "#FFFBEB", "#92400E"
            else:
                badge_bg, badge_color = "#FEF2F2", "#991B1B"

            bar_w = int(h['conf'])
            bar_color = "#12B76A" if h['conf'] >= 90 else "#F59E0B" if h['conf'] >= 70 else "#EF4444"

            st.markdown(f"""
            <div style="
              background:#FFFFFF; border:1px solid #E4E8EF; border-radius:10px;
              padding:10px 12px; margin-bottom:8px;
              box-shadow:0 1px 3px rgba(15,21,35,.04);
              transition: box-shadow .2s;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
                <span style="
                  font-family:'Poppins',sans-serif; font-size:28px;
                  font-weight:700; color:#0F1523; line-height:1;">
                  {h['digit']}
                </span>
                <span style="
                  font-family:'Poppins',sans-serif; font-size:11px; font-weight:600;
                  background:{badge_bg}; color:{badge_color};
                  border-radius:6px; padding:2px 8px;">
                  {h['conf']}%
                </span>
              </div>
              <div style="background:#F0F2F5; border-radius:99px; height:3px; overflow:hidden;">
                <div style="
                  width:{bar_w}%; height:100%;
                  background:{bar_color}; border-radius:99px;
                  transition: width .4s ease;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="
          background:#F7F8FA; border:1px dashed #E4E8EF; border-radius:10px;
          padding:28px 16px; text-align:center; margin-top:4px;">
          <div style="font-size:28px; margin-bottom:8px; opacity:.4;">📋</div>
          <div style="font-family:'Poppins',sans-serif; font-size:12px;
                      font-weight:500; color:#9CA3AF;">
            No predictions yet
          </div>
          <div style="font-family:'Poppins',sans-serif; font-size:11px;
                      color:#C0C6D0; margin-top:4px;">
            Upload an image or draw a digit to get started
          </div>
        </div>
        """, unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("""
    <div style="
      font-family:'Poppins',sans-serif; font-size:14px; font-weight:600;
      color:#0F1523; margin-bottom:16px; display:flex; align-items:center; gap:8px;">
      <div style="
        width:6px; height:20px; background:#3B6EF8;
        border-radius:3px; flex-shrink:0;"></div>
      Input
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_draw = st.tabs(["Upload Image", "Draw Digit"])

    result = None

    with tab_upload:
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop an image here, or click to browse",
            type=["png", "jpg", "jpeg", "bmp", "gif"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read()))
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded image", use_column_width=True)
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            if st.button("Analyze Image", key="analyze_upload"):
                with st.spinner("Running inference…"):
                    arr = preprocess_upload(img)
                    preview_img = Image.fromarray(
                        (arr.reshape(28, 28) * 255).astype(np.uint8), mode="L"
                    )
                    st.session_state.preview = preview_img
                    result = predict_from_array(arr)
        else:
            st.markdown("""
            <div style="
              background:#F7F8FA; border:2px dashed #E4E8EF; border-radius:12px;
              padding:36px 20px; text-align:center; margin-top:12px;">
              <div style="font-size:32px; margin-bottom:10px; opacity:.5;">🖼</div>
              <div style="font-family:'Poppins',sans-serif; font-size:13px;
                          font-weight:500; color:#5A6478; margin-bottom:4px;">
                Drop your image here
              </div>
              <div style="font-family:'Poppins',sans-serif; font-size:11px; color:#9CA3AF;">
                Supports PNG, JPG, BMP, GIF
              </div>
            </div>
            """, unsafe_allow_html=True)

    from streamlit_drawable_canvas import st_canvas

    with tab_draw:
        st.markdown("""
        <div style="
          font-family:'Poppins',sans-serif; font-size:12px; font-weight:400;
          color:#5A6478; margin: 12px 0 10px; line-height:1.5;">
          Draw a digit (0–9) below. Prediction updates <strong>automatically</strong> when you lift your pen.
        </div>
        """, unsafe_allow_html=True)

        if "canvas_key"  not in st.session_state:
            st.session_state.canvas_key  = 0
        if "eraser_on"   not in st.session_state:
            st.session_state.eraser_on   = False
        if "stroke_history" not in st.session_state:
            st.session_state.stroke_history = []
        if "redo_stack" not in st.session_state:
            st.session_state.redo_stack = []

        st.markdown("""
        <style>
        /* Scope ONLY to draw-toolbar so global .stButton override doesn't win */
        .draw-toolbar-wrap button {
            font-family: 'Poppins', sans-serif !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            width: 100%;
            cursor: pointer;
            border: none;
            transition: all .2s;
        }
        /* Eraser — inactive */
        .eraser-inactive button {
            background: #F0F2F5 !important;
            color: #5A6478 !important;
            border: 1.5px solid #E4E8EF !important;
            box-shadow: none !important;
            transform: none !important;
        }
        .eraser-inactive button:hover {
            background: #E4E8EF !important;
            color: #0F1523 !important;
            transform: none !important;
            box-shadow: none !important;
        }
        /* Eraser — active */
        .eraser-active button {
            background: #EEF2FF !important;
            color: #3B6EF8 !important;
            border: 1.5px solid #3B6EF8 !important;
            box-shadow: none !important;
            transform: none !important;
        }
        /* Undo */
        .undo-btn button {
            background: #F0F2F5 !important;
            color: #5A6478 !important;
            border: 1.5px solid #E4E8EF !important;
            box-shadow: none !important;
            transform: none !important;
        }
        .undo-btn button:hover {
            background: #E4E8EF !important;
            color: #0F1523 !important;
            box-shadow: none !important;
            transform: none !important;
        }
        /* Redo */
        .redo-btn button {
            background: #F0F2F5 !important;
            color: #5A6478 !important;
            border: 1.5px solid #E4E8EF !important;
            box-shadow: none !important;
            transform: none !important;
        }
        .redo-btn button:hover {
            background: #E4E8EF !important;
            color: #0F1523 !important;
            box-shadow: none !important;
            transform: none !important;
        }
        /* Clear */
        .clear-btn button {
            background: #FEF2F2 !important;
            color: #EF4444 !important;
            border: 1.5px solid #FECACA !important;
            box-shadow: none !important;
            transform: none !important;
        }
        .clear-btn button:hover {
            background: #FEE2E2 !important;
            box-shadow: none !important;
            transform: none !important;
        }

        /* Remove the black gap — hide the extra div st_canvas injects */
        [data-testid="stCustomComponentV1"] > div > div {
            background: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)

        c_eraser, c_undo, c_redo, c_clear = st.columns([2, 1, 1, 1])

        eraser_class = "eraser-active" if st.session_state.eraser_on else "eraser-inactive"

        with c_eraser:
            eraser_label = "◻  Eraser  (ON)" if st.session_state.eraser_on else "◻  Eraser"
            st.markdown(f'<div class="draw-toolbar-wrap {eraser_class}">', unsafe_allow_html=True)
            if st.button(eraser_label, use_container_width=True, key="eraser_btn"):
                st.session_state.eraser_on = not st.session_state.eraser_on
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c_undo:
            st.markdown('<div class="draw-toolbar-wrap undo-btn">', unsafe_allow_html=True)
            undo_clicked = st.button("↩  Undo", use_container_width=True, key="undo_btn")
            st.markdown('</div>', unsafe_allow_html=True)

        with c_redo:
            st.markdown('<div class="draw-toolbar-wrap redo-btn">', unsafe_allow_html=True)
            redo_clicked = st.button("↪  Redo", use_container_width=True, key="redo_btn")
            st.markdown('</div>', unsafe_allow_html=True)

        with c_clear:
            st.markdown('<div class="draw-toolbar-wrap clear-btn">', unsafe_allow_html=True)
            if st.button("🗑  Clear", use_container_width=True, key="clear_btn"):
                st.session_state.canvas_key += 1
                st.session_state.eraser_on  = False
                st.session_state.preview    = None
                st.session_state.stroke_history = []
                st.session_state.redo_stack = []
                result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if undo_clicked:
            if st.session_state.stroke_history:
                current = st.session_state.stroke_history.pop()
                st.session_state.redo_stack.append(current)
            st.session_state.canvas_key += 1
            st.rerun()

        if redo_clicked:
            if st.session_state.redo_stack:
                restored = st.session_state.redo_stack.pop()
                st.session_state.stroke_history.append(restored)
            st.session_state.canvas_key += 1
            st.rerun()

        stroke_color = "#000000" if st.session_state.eraser_on else "#FFFFFF"
        stroke_width = 30        if st.session_state.eraser_on else 18

        initial_drawing = None
        if st.session_state.stroke_history:
            initial_drawing = st.session_state.stroke_history[-1]

        canvas_result = st_canvas(
            fill_color       = "rgba(0,0,0,0)",
            stroke_width     = stroke_width,
            stroke_color     = stroke_color,
            background_color = "#000000",
            height           = 280,
            width            = 280,
            drawing_mode     = "freedraw",
            key              = f"canvas_{st.session_state.canvas_key}",
            display_toolbar  = False,
            update_streamlit = True,
            initial_drawing  = initial_drawing,
        )

        if (canvas_result.json_data is not None
                and canvas_result.json_data.get("objects")):
            current_objects = canvas_result.json_data["objects"]
            prev_count = len(st.session_state.stroke_history[-1]["objects"]) if st.session_state.stroke_history else 0
            if len(current_objects) != prev_count:
                st.session_state.stroke_history.append(canvas_result.json_data)
                st.session_state.redo_stack = []

        MIN_WHITE_PIXELS = 500

        if canvas_result.image_data is not None:
            white_pixels = (canvas_result.image_data[:, :, :3] > 128).sum()
            if white_pixels >= MIN_WHITE_PIXELS:
                arr = preprocess_canvas(canvas_result.image_data)
                preview_img = Image.fromarray(
                    (arr.reshape(28, 28) * 255).astype(np.uint8), mode="L"
                )
                st.session_state.preview = preview_img
                with st.spinner("Running inference…"):
                    result = predict_from_array(arr)
with col_output:
    st.markdown("""
    <div style="
      font-family:'Poppins',sans-serif; font-size:14px; font-weight:600;
      color:#0F1523; margin-bottom:16px; display:flex; align-items:center; gap:8px;">
      <div style="
        width:6px; height:20px; background:#3B6EF8;
        border-radius:3px; flex-shrink:0;"></div>
      Prediction
    </div>
    """, unsafe_allow_html=True)

    if result:
        digit, conf, probs, ms, top3 = result

        st.session_state.history.append({"digit": digit, "conf": conf})

        if conf < 60:
            st.markdown("""
            <div style="
              background:#FEF2F2; border:1px solid #FECACA; border-radius:10px;
              padding:10px 14px; margin-bottom:14px; display:flex; align-items:center; gap:10px;">
              <span style="font-size:16px;">⚠️</span>
              <div>
                <div style="font-family:'Poppins',sans-serif; font-size:12px;
                            font-weight:600; color:#991B1B;">Low Confidence</div>
                <div style="font-family:'Poppins',sans-serif; font-size:11px;
                            color:#B91C1C;">Try redrawing the digit larger and centered.</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        elif conf < 80:
            st.markdown("""
            <div style="
              background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px;
              padding:10px 14px; margin-bottom:14px; display:flex; align-items:center; gap:10px;">
              <span style="font-size:16px;">💡</span>
              <div>
                <div style="font-family:'Poppins',sans-serif; font-size:12px;
                            font-weight:600; color:#92400E;">Moderate Confidence</div>
                <div style="font-family:'Poppins',sans-serif; font-size:11px;
                            color:#B45309;">Drawing more clearly may improve accuracy.</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if conf >= 90:
            result_color = "#12B76A"
            result_bg = "#ECFDF5"
            result_border = "#A7F3D0"
        elif conf >= 70:
            result_color = "#F59E0B"
            result_bg = "#FFFBEB"
            result_border = "#FDE68A"
        else:
            result_color = "#EF4444"
            result_bg = "#FEF2F2"
            result_border = "#FECACA"

        st.markdown(f"""
        <div style="
          background:var(--bg-surface,#fff); border:1px solid var(--border,#E4E8EF);
          border-radius:16px; padding:28px 24px; text-align:center;
          margin-bottom:16px; box-shadow:0 4px 16px rgba(15,21,35,.06);">

          <div style="
            display:inline-flex; align-items:center; gap:6px;
            background:{result_bg}; border:1px solid {result_border};
            border-radius:99px; padding:4px 12px; margin-bottom:16px;">
            <div style="
              width:6px;height:6px;border-radius:50%;
              background:{result_color};"></div>
            <span style="
              font-family:'Poppins',sans-serif; font-size:11px; font-weight:600;
              color:{result_color}; letter-spacing:.03em;">
              {conf}% Confidence
            </span>
          </div>

          <div style="
            font-family:'Poppins',sans-serif; font-size:96px;
            font-weight:700; line-height:1; color:#0F1523;
            letter-spacing:-.04em; margin-bottom:12px;">
            {digit}
          </div>

          <div style="
            font-family:'Poppins',sans-serif; font-size:12px; font-weight:500;
            color:#9CA3AF; letter-spacing:.02em;">
            Recognized Digit
          </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Digit", str(digit))
        with m2: st.metric("Confidence", f"{conf}%")
        with m3: st.metric("Inference", f"{ms} ms")

        st.markdown("""
        <div style="
          font-family:'Poppins',sans-serif; font-size:11px; font-weight:600;
          color:#9CA3AF; text-transform:uppercase; letter-spacing:.06em;
          margin: 16px 0 6px;">
          Confidence Score
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf))

        st.markdown("""
        <div style="
          font-family:'Poppins',sans-serif; font-size:11px; font-weight:600;
          color:#9CA3AF; text-transform:uppercase; letter-spacing:.06em;
          margin: 20px 0 10px;">
          Top 3 Predictions
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3 = st.columns(3)
        rank_styles = [
            ("#EEF2FF", "#3B6EF8", "#3B6EF8"),
            ("#F7F8FA", "#5A6478", "#9CA3AF"),
            ("#F7F8FA", "#5A6478", "#9CA3AF"),
        ]
        for col, (d, c), (bg, num_color, pct_color) in zip([t1, t2, t3], top3, rank_styles):
            with col:
                st.markdown(f"""
                <div style="
                  background:{bg}; border:1px solid #E4E8EF;
                  border-radius:10px; padding:12px 8px; text-align:center;">
                  <div style="
                    font-family:'Poppins',sans-serif; font-size:30px;
                    font-weight:700; color:{num_color}; line-height:1;">{d}</div>
                  <div style="
                    font-family:'Poppins',sans-serif; font-size:11px;
                    font-weight:600; color:{pct_color}; margin-top:4px;">{c}%</div>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.preview:
            st.markdown("""
            <div style="
              font-family:'Poppins',sans-serif; font-size:11px; font-weight:600;
              color:#9CA3AF; text-transform:uppercase; letter-spacing:.06em;
              margin: 20px 0 10px;">
              Model Input Preview (28 × 28)
            </div>
            """, unsafe_allow_html=True)
            st.image(st.session_state.preview, width=112,
                     caption="Downscaled to 28×28 before inference")

    else:
          import base64
          _svg = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2V6"                   stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M12 18V22"                 stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M4.93 4.93L7.76 7.76"     stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M16.24 16.24L19.07 19.07" stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M2 12H6"                   stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M18 12H22"                 stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M4.93 19.07L7.76 16.24"   stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <path d="M16.24 7.76L19.07 4.93"   stroke="#4F46E5" stroke-width="2" stroke-linecap="round"/>
    <circle cx="12" cy="12" r="3" fill="#4F46E5"/>
  </svg>"""
          _icon = "data:image/svg+xml;base64," + base64.b64encode(_svg.encode()).decode()

          st.markdown(f"""
  <div style="
    background:#FFFFFF;
    border:1px solid #E4E8EF;
    border-radius:16px;
    padding:60px 32px;
    text-align:center;
    box-shadow:0 1px 4px rgba(15,21,35,.04);">

    <div style="
      width:56px;
      height:56px;
      border-radius:14px;
      background:#F0F2F5;
      line-height:56px;
      margin:0 auto 16px;">
      <img src="{_icon}" width="28" height="28"
          style="vertical-align:middle;" alt="icon"/>
    </div>

    <div style="
      font-family:'Poppins',sans-serif;
      font-size:14px;
      font-weight:600;
      color:#5A6478;
      margin-bottom:6px;">
      Awaiting Input
    </div>

    <div style="
      font-family:'Poppins',sans-serif;
      font-size:12px;
      color:#9CA3AF;
      max-width:220px;
      margin:0 auto;
      line-height:1.6;">
      Upload an image or draw a digit on the left to see the prediction here.
    </div>

  </div>
  """, unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("""
<div style="
  display:flex; align-items:center; gap:8px;
  font-family:'Poppins',sans-serif; font-size:14px; font-weight:600;
  color:#0F1523; margin-bottom:16px;">
  <div style="width:6px;height:20px;background:#3B6EF8;border-radius:3px;flex-shrink:0;"></div>
  Class Probability Distribution
</div>
""", unsafe_allow_html=True)

if result:
    digit, conf, probs, ms, top3 = result
    cols = st.columns(10)
    for i, col in enumerate(cols):
        pct = round(probs[i] * 100, 1)
        is_top = (i == digit)
        if is_top:
            bg      = "#EEF2FF"
            border  = "#C7D7FD"
            num_c   = "#3B6EF8"
            pct_c   = "#3B6EF8"
            bar_c   = "#3B6EF8"
        else:
            bg      = "#FFFFFF"
            border  = "#E4E8EF"
            num_c   = "#0F1523"
            pct_c   = "#9CA3AF"
            bar_c   = "#CBD5E1"

        with col:
            st.markdown(f"""
            <div style="
              background:{bg}; border:1px solid {border};
              border-radius:10px; padding:10px 4px; text-align:center;
              box-shadow:0 1px 3px rgba(15,21,35,.04);
              transition:box-shadow .2s;">
              <div style="
                font-family:'Poppins',sans-serif; font-size:20px;
                font-weight:700; color:{num_c}; line-height:1;">{i}</div>
              <div style="
                font-family:'Poppins',sans-serif; font-size:9px;
                font-weight:600; color:{pct_c}; margin-top:4px;">{pct}%</div>
              <div style="
                background:#F0F2F5; border-radius:99px; height:4px;
                margin-top:8px; overflow:hidden;">
                <div style="
                  width:{pct}%; height:100%;
                  background:{bar_c}; border-radius:99px;
                  transition:width .4s ease;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="
      background:#FFFFFF; border:1px solid #E4E8EF; border-radius:12px;
      padding:28px; text-align:center; box-shadow:0 1px 3px rgba(15,21,35,.04);">
      <div style="
        font-family:'Poppins',sans-serif; font-size:12px; font-weight:500;
        color:#9CA3AF;">
        Run an analysis to see the full probability distribution across all digit classes.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("""
<div style="
  display:flex; align-items:center; gap:8px;
  font-family:'Poppins',sans-serif; font-size:14px; font-weight:600;
  color:#0F1523; margin-bottom:16px;">
  <div style="width:6px;height:20px;background:#3B6EF8;border-radius:3px;flex-shrink:0;"></div>
  Model Specifications
</div>
""", unsafe_allow_html=True)

specs = [
    ("CNN", "Architecture"),
    ("5", "Epochs"),
    ("60,000", "Train Images"),
    ("10,000", "Test Images"),
    ("99.2%", "Accuracy"),
    ("28 × 28", "Input Shape"),
    ("Adam", "Optimizer"),
    ("10", "Output Classes"),
]
scols = st.columns(8)
for col, (val, key) in zip(scols, specs):
    with col:
        st.metric(key, val)

