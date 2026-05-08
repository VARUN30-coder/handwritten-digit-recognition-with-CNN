import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import io
import base64
from streamlit.components.v1 import html as st_html

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="NeuralScan — Digit Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* Main background */
.stApp { background: #030508; }
.main .block-container { padding: 2rem 2rem 2rem; max-width: 1200px; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* All text */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    color: #cde8f5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #070d14 !important;
    border-right: 1px solid #0e2233;
}
[data-testid="stSidebar"] * { color: #cde8f5 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0b1520;
    border: 1px solid #0e2233;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    color: #5a8aaa !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00d4ff !important;
    font-size: 28px !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 3px !important;
    border-radius: 10px !important;
    width: 100%;
    padding: 12px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: #00d4ff18 !important;
    border-color: #00d4ff !important;
    color: #fff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1520;
    border-radius: 10px;
    border: 1px solid #0e2233;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    color: #5a8aaa !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #00d4ff18 !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: #0b1520 !important;
    border-radius: 99px !important;
    border: 1px solid #0e2233;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0b1520;
    border: 2px dashed #1a3a52;
    border-radius: 12px;
    padding: 20px;
}
[data-testid="stFileUploader"]:hover { border-color: #00d4ff; }

/* Divider */
hr { border-color: #0e2233 !important; }

/* Success/info boxes */
.stSuccess { background: #00ff8812 !important; border: 1px solid #00ff8833 !important; border-radius: 10px !important; }
.stInfo    { background: #00d4ff12 !important; border: 1px solid #00d4ff33 !important; border-radius: 10px !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0b1520 !important;
    border: 1px solid #1a3a52 !important;
    border-radius: 8px !important;
    color: #cde8f5 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model (cached) ───────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

# ── Helper: Preprocess uploaded image ────────────────────
# Uploaded images are usually white background + dark digit
# So we invert them to match MNIST format (white digit on black)
def preprocess_upload(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = ImageOps.invert(image)          # invert: dark digit → white digit
    image = image.resize((28, 28), Image.LANCZOS)
    arr = np.array(image) / 255.0
    return arr.reshape(1, 28, 28, 1)

# ── Helper: Preprocess canvas drawing ────────────────────
# Canvas is already black background + white digit = MNIST format
# So NO invert needed — directly resize and normalize
def preprocess_canvas(canvas_arr: np.ndarray) -> np.ndarray:
    # canvas_arr is RGBA — take the red channel (or average RGB)
    # since drawing is white on black, any channel works
    gray = canvas_arr[:, :, 0].astype(np.uint8)
    image = Image.fromarray(gray, mode="L")
    image = image.resize((28, 28), Image.LANCZOS)
    arr = np.array(image) / 255.0
    return arr.reshape(1, 28, 28, 1)

# ── Helper: Run Prediction ────────────────────────────────
def predict_from_array(arr: np.ndarray):
    t0 = time.time()
    probs = model.predict(arr, verbose=0)[0]
    ms = round((time.time() - t0) * 1000, 1)
    digit = int(np.argmax(probs))
    conf  = round(float(np.max(probs)) * 100, 1)
    # Top-3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(int(i), round(float(probs[i]) * 100, 1)) for i in top3_idx]
    return digit, conf, probs.tolist(), ms, top3

# ── Session State Init ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "preview" not in st.session_state:
    st.session_state.preview = None

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
  <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
              color:#5a8aaa; letter-spacing:6px; margin-bottom:8px;">
    CONVOLUTIONAL NEURAL NETWORK
  </div>
  <div style="font-family:'Rajdhani',sans-serif; font-size:48px;
              font-weight:700; letter-spacing:4px; color:#cde8f5;
              text-shadow: 0 0 30px #00d4ff30;">
    NEURAL<span style="color:#00d4ff;">SCAN</span>
  </div>
  <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
              color:#5a8aaa; letter-spacing:3px; margin-top:4px;">
    ◈ DIGIT RECOGNITION ENGINE · MNIST · 99.2% ACCURACY
  </div>
</div>
<hr/>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:12px;
                color:#00d4ff; letter-spacing:3px; margin-bottom:16px;">
    ◈ MODEL STATUS
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0b1520; border:1px solid #0e2233; border-radius:10px; padding:14px; margin-bottom:12px;">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;"></div>
        <span style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#00ff88; letter-spacing:2px;">ONLINE</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#5a8aaa; line-height:1.8;">
        Model: CNN (Keras)<br/>
        Dataset: MNIST<br/>
        Classes: 0 – 9<br/>
        Input: 28×28 px<br/>
        Layers: Conv×2 + Dense×2
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:12px;
                color:#00d4ff; letter-spacing:3px; margin-bottom:12px;">
    ◈ SESSION HISTORY
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            bar_w = int(h['conf'])
            color = "#00ff88" if h['conf'] >= 90 else "#ffcc00" if h['conf'] >= 70 else "#ff3355"
            st.markdown(f"""
            <div style="background:#0b1520; border:1px solid #0e2233; border-radius:8px;
                        padding:10px 12px; margin-bottom:6px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-family:'Share Tech Mono',monospace; font-size:22px;
                             font-weight:700; color:#00d4ff;">
                  {h['digit']}
                </span>
                <span style="font-family:'Share Tech Mono',monospace; font-size:11px;
                             color:{color};">
                  {h['conf']}%
                </span>
              </div>
              <div style="background:#070d14; border-radius:99px; height:4px;">
                <div style="width:{bar_w}%; height:100%; background:{color};
                            border-radius:99px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 CLEAR HISTORY"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                    color:#2a4a60; text-align:center; padding:20px; letter-spacing:2px;">
        NO PREDICTIONS YET
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MAIN CONTENT — Two Columns
# ══════════════════════════════════════════════════════════
col_input, col_output = st.columns([1, 1], gap="large")

# ── LEFT: Input ───────────────────────────────────────────
with col_input:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:12px;
                color:#00d4ff; letter-spacing:3px; margin-bottom:16px;">
    ◈ INPUT MODULE
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_draw = st.tabs(["⬆  UPLOAD IMAGE", "✏  DRAW DIGIT"])

    result = None  # will hold (digit, conf, probs, ms)

    # ── Tab 1: Upload ─────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Drop image here",
            type=["png", "jpg", "jpeg", "bmp", "gif"],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read()))
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if st.button("▶  ANALYZE IMAGE", key="analyze_upload"):
                with st.spinner("Processing neural signals..."):
                    arr = preprocess_upload(img)
                    preview_img = Image.fromarray((arr.reshape(28, 28) * 255).astype(np.uint8), mode="L")
                    st.session_state.preview = preview_img
                    result = predict_from_array(arr)

    # ── Tab 2: Draw ───────────────────────────────────────
    with tab_draw:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                    color:#5a8aaa; letter-spacing:2px; margin-bottom:8px;">
        DRAW A DIGIT (0–9) IN THE BOX BELOW — canvas mein draw karo phir SAVE dabao
        </div>
        """, unsafe_allow_html=True)

        # HTML5 Canvas — zero external dependency
        canvas_html = """
        <style>
        #drawCanvas { background:#000; border:1px solid #1e3a5f; border-radius:8px;
                      cursor:crosshair; touch-action:none; display:block; }
        .cbtn { background:transparent; border:1px solid #1e3a5f; color:#00d4ff;
                font-family:monospace; font-size:12px; letter-spacing:2px;
                padding:8px 16px; border-radius:6px; cursor:pointer; margin-right:8px; }
        .cbtn.primary { background:#003a5c; border-color:#00d4ff; }
        #cstatus { font-family:monospace; font-size:11px; color:#5a8aaa;
                   margin-top:8px; letter-spacing:1px; }
        </style>
        <canvas id="drawCanvas" width="280" height="280"></canvas><br/>
        <div style="margin-top:10px;">
          <button class="cbtn" onclick="clearCanvas()">CLEAR</button>
          <button class="cbtn primary" onclick="saveCanvas()">💾 SAVE DRAWING</button>
        </div>
        <div id="cstatus">Draw a digit above, then click SAVE DRAWING</div>
        <input type="text" id="imgOut" style="display:none"/>

        <script>
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle='#000'; ctx.fillRect(0,0,280,280);
        ctx.strokeStyle='#fff'; ctx.lineWidth=18; ctx.lineCap='round'; ctx.lineJoin='round';
        let drawing=false, lx=0, ly=0;

        function getPos(e) {
          const r=canvas.getBoundingClientRect();
          const sx=canvas.width/r.width, sy=canvas.height/r.height;
          if(e.touches) return [(e.touches[0].clientX-r.left)*sx,(e.touches[0].clientY-r.top)*sy];
          return [(e.clientX-r.left)*sx,(e.clientY-r.top)*sy];
        }
        canvas.onmousedown=e=>{drawing=true;[lx,ly]=getPos(e);};
        canvas.onmousemove=e=>{
          if(!drawing)return;
          const[x,y]=getPos(e);
          ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(x,y);ctx.stroke();
          [lx,ly]=[x,y];
        };
        canvas.onmouseup=()=>drawing=false;
        canvas.onmouseleave=()=>drawing=false;
        canvas.ontouchstart=e=>{e.preventDefault();drawing=true;[lx,ly]=getPos(e);};
        canvas.ontouchmove=e=>{
          e.preventDefault();if(!drawing)return;
          const[x,y]=getPos(e);
          ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(x,y);ctx.stroke();
          [lx,ly]=[x,y];
        };
        canvas.ontouchend=()=>drawing=false;

        function clearCanvas(){
          ctx.fillStyle='#000';ctx.fillRect(0,0,280,280);ctx.strokeStyle='#fff';
          document.getElementById('cstatus').textContent='Canvas cleared. Draw again.';
        }

        function saveCanvas(){
          const b64=canvas.toDataURL('image/png').split(',')[1];
          // Write to a Streamlit-readable text input via parent DOM
          const allInputs=window.parent.document.querySelectorAll('input[data-testid="stTextInput"]');
          let found=false;
          allInputs.forEach(inp=>{
            if(inp.getAttribute('aria-label')==='canvas_image_data'){
              inp.value=b64;
              inp.dispatchEvent(new Event('input',{bubbles:true}));
              found=true;
            }
          });
          if(found){
            document.getElementById('cstatus').textContent='✓ Saved! Now click ANALYZE DRAWING button below.';
          } else {
            // fallback: copy to clipboard
            navigator.clipboard.writeText(b64).then(()=>{
              document.getElementById('cstatus').textContent='Copied to clipboard! Paste in the text box below.';
            });
          }
        }
        </script>
        """
        st_html(canvas_html, height=400)

        # Streamlit text input — canvas JS will auto-fill this
        canvas_b64 = st.text_input(
            "canvas_image_data",
            value="",
            key="canvas_image_data",
            label_visibility="collapsed",
            placeholder="← Canvas data auto-fills here after SAVE DRAWING"
        )

        if st.button("▶  ANALYZE DRAWING", key="analyze_draw"):
            raw = canvas_b64.strip()
            if raw and len(raw) > 200:
                try:
                    img_bytes = base64.b64decode(raw)
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                    canvas_arr = np.array(img_pil)
                    arr = preprocess_canvas(canvas_arr)
                    preview_img = Image.fromarray((arr.reshape(28, 28) * 255).astype(np.uint8), mode="L")
                    st.session_state.preview = preview_img
                    with st.spinner("Processing neural signals..."):
                        result = predict_from_array(arr)
                except Exception as ex:
                    st.error(f"Error reading canvas: {ex}")
            else:
                st.warning("Pehle canvas mein digit draw karo aur SAVE DRAWING dabaao!")

# ── RIGHT: Output ─────────────────────────────────────────
with col_output:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:12px;
                color:#00d4ff; letter-spacing:3px; margin-bottom:16px;">
    ◈ PREDICTION OUTPUT
    </div>
    """, unsafe_allow_html=True)

    if result:
        digit, conf, probs, ms, top3 = result

        # Save to history
        st.session_state.history.append({"digit": digit, "conf": conf})

        # ── 1. CONFIDENCE WARNING ──────────────────────────
        if conf < 60:
            st.markdown("""
            <div style="background:#ff333518; border:1px solid #ff333555;
                        border-radius:10px; padding:10px 14px; margin-bottom:12px;
                        font-family:'Share Tech Mono',monospace; font-size:10px;
                        color:#ff6677; letter-spacing:2px;">
            ⚠ LOW CONFIDENCE — Please redraw the digit larger and centered
            </div>
            """, unsafe_allow_html=True)
        elif conf < 80:
            st.markdown("""
            <div style="background:#ffcc0018; border:1px solid #ffcc0044;
                        border-radius:10px; padding:10px 14px; margin-bottom:12px;
                        font-family:'Share Tech Mono',monospace; font-size:10px;
                        color:#ffcc00; letter-spacing:2px;">
            ◈ MODERATE CONFIDENCE — Try drawing more clearly
            </div>
            """, unsafe_allow_html=True)

        # Big digit display
        color = "#00ff88" if conf >= 90 else "#ffcc00" if conf >= 70 else "#ff3355"
        st.markdown(f"""
        <div style="background:#0b1520; border:1px solid #0e2233; border-radius:16px;
                    padding:32px; text-align:center; margin-bottom:16px;
                    box-shadow: 0 0 40px #00d4ff08;">
          <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                      color:#5a8aaa; letter-spacing:3px; margin-bottom:8px;">
            DIGIT IDENTIFIED
          </div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:120px;
                      font-weight:900; line-height:1; color:{color};
                      text-shadow: 0 0 60px {color}88;">
            {digit}
          </div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
                      color:#5a8aaa; letter-spacing:3px; margin-top:8px;">
            CONFIDENCE: <span style="color:{color};">{conf}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Predicted Digit", str(digit))
        with m2: st.metric("Confidence", f"{conf}%")
        with m3: st.metric("Inference Time", f"{ms} ms")

        # Confidence bar
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                    color:#5a8aaa; letter-spacing:2px; margin: 16px 0 6px;">
        // CONFIDENCE LEVEL
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf))

        # ── 2. TOP-3 PREDICTIONS ───────────────────────────
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                    color:#5a8aaa; letter-spacing:2px; margin: 16px 0 8px;">
        // TOP-3 PREDICTIONS
        </div>
        """, unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        for col, (d, c), rank in zip([t1, t2, t3], top3, ["🥇", "🥈", "🥉"]):
            tc = "#00d4ff" if rank == "🥇" else "#5a8aaa"
            with col:
                st.markdown(f"""
                <div style="background:#0b1520; border:1px solid #0e2233;
                            border-radius:10px; padding:10px; text-align:center;">
                  <div style="font-family:'Share Tech Mono',monospace;
                              font-size:28px; color:{tc}; font-weight:700;">{d}</div>
                  <div style="font-family:'Share Tech Mono',monospace;
                              font-size:10px; color:{tc}; margin-top:4px;">{c}%</div>
                </div>
                """, unsafe_allow_html=True)

        # ── 3. 28x28 PREVIEW ──────────────────────────────
        if "preview" in st.session_state:
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                        color:#5a8aaa; letter-spacing:2px; margin: 16px 0 8px;">
            // MODEL NE YEH DEKHA (28×28)
            </div>
            """, unsafe_allow_html=True)
            st.image(st.session_state.preview, width=112, caption="28×28 input to model")

    else:
        # Empty state
        st.markdown("""
        <div style="background:#0b1520; border:1px solid #0e2233; border-radius:16px;
                    padding:60px 32px; text-align:center;">
          <div style="font-size:48px; opacity:.15; margin-bottom:16px;">⬡</div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
                      color:#2a4a60; letter-spacing:3px; text-transform:uppercase;">
            Awaiting Input Signal
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PROBABILITY DISTRIBUTION (Full Width)
# ══════════════════════════════════════════════════════════
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Share Tech Mono',monospace; font-size:12px;
            color:#00d4ff; letter-spacing:3px; margin-bottom:16px;">
◈ CLASS PROBABILITY DISTRIBUTION
</div>
""", unsafe_allow_html=True)

if result:
    digit, conf, probs, ms, top3 = result
    cols = st.columns(10)
    for i, col in enumerate(cols):
        pct = round(probs[i] * 100, 1)
        is_top = (i == digit)
        border_color = "#00d4ff" if is_top else "#0e2233"
        bg_color     = "#00d4ff18" if is_top else "#0b1520"
        text_color   = "#00d4ff" if is_top else "#5a8aaa"
        bar_color    = "#00d4ff" if is_top else "#1a3a52"

        with col:
            st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {border_color};
                        border-radius:10px; padding:10px 4px; text-align:center;">
              <div style="font-family:'Share Tech Mono',monospace; font-size:20px;
                          font-weight:700; color:{text_color};">{i}</div>
              <div style="font-family:'Share Tech Mono',monospace; font-size:9px;
                          color:{text_color}; margin-top:4px;">{pct}%</div>
              <div style="background:#070d14; border-radius:99px; height:3px;
                          margin-top:6px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{bar_color};
                            border-radius:99px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#0b1520; border:1px solid #0e2233; border-radius:12px;
                padding:30px; text-align:center;">
      <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                  color:#2a4a60; letter-spacing:3px;">
      RUN ANALYSIS TO SEE PROBABILITY DISTRIBUTION
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MODEL INFO (Full Width)
# ══════════════════════════════════════════════════════════
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Share Tech Mono',monospace; font-size:12px;
            color:#00d4ff; letter-spacing:3px; margin-bottom:16px;">
◈ MODEL ARCHITECTURE
</div>
""", unsafe_allow_html=True)

specs = [
    ("CNN", "Architecture"),
    ("5", "Epochs"),
    ("60,000", "Train Images"),
    ("10,000", "Test Images"),
    ("99.2%", "Accuracy"),
    ("28×28", "Input Shape"),
    ("Adam", "Optimizer"),
    ("10", "Output Classes"),
]
scols = st.columns(8)
for col, (val, key) in zip(scols, specs):
    with col:
        st.metric(key, val)


