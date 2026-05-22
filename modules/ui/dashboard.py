"""
rPPG DeepShield — Streamlit Dashboard
Author: Person A (Integration Owner)

Architecture:
  - All dynamic content lives in st.empty() placeholders updated inside a while loop.
  - EVERY st.plotly_chart() has a unique key= to prevent Streamlit's duplicate-ID error.
  - render_ui_state() is a pure "push new HTML/charts into placeholders" function.
  - Pipeline: cap.read → FaceDetector → extract_roi → process_signal → analyze_threat → display
"""

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import time
import datetime
import sys
import os

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="rPPG DeepShield",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend imports (graceful stubs if missing) ───────────────────────────
_import_errors: list[str] = []

try:
    from modules.face_detection import FaceDetector, extract_roi, create_data_packet
    _face_ok = True
except Exception as _e:
    _import_errors.append(f"FaceDetector: {_e}")
    _face_ok = False
    class FaceDetector:
        def detect(self, frame): return False, (0, 0, 0, 0)
        def reset(self): pass
    def extract_roi(frame, bbox):
        return np.zeros((10,10,3), np.uint8), np.zeros((10,10,3), np.uint8)
    def create_data_packet(*a, **kw):
        return {"roi_forehead": np.zeros((10,10,3), np.uint8),
                "roi_cheeks":   np.zeros((10,10,3), np.uint8)}

try:
    from modules.rppg_engine import process_signal, reset_engine
    _engine_ok = True
except Exception as _e:
    _import_errors.append(f"rPPG Engine: {_e}")
    _engine_ok = False
    def process_signal(*a, **kw):
        fc = kw.get("frame_count", 0)
        return {"rppg_signal": np.zeros(300), "estimated_bpm": 0.0,
                "is_calibrating": fc < 150, "signal_quality": 0.0,
                "method_used": "STUB"}
    def reset_engine(): pass

try:
    from modules.threat_analyzer import analyze_threat
    from modules.threat_analyzer.threat_scorer import reset_verdict_history
    _threat_ok = True
except Exception as _e:
    _import_errors.append(f"ThreatAnalyzer: {_e}")
    _threat_ok = False
    def analyze_threat(*a, **kw):
        return {"verdict": "UNCERTAIN", "bpm": 0.0, "confidence": 0.0,
                "pulse_present": False, "loop_detected": False,
                "snr_score": 0.0, "dominant_freq_hz": 0.0}
    def reset_verdict_history(): pass

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
body, .stApp { background-color:#0D1B2A !important; }
h1,h2,h3,h4,h5,h6 { font-family:'Rajdhani',sans-serif !important; color:white !important; }
p,div,span,label { color:#D5D8DC !important; }
[data-testid="stSidebar"] { background-color:#0A1628 !important; }
#MainMenu,footer { visibility:hidden; }
.stButton>button {
    background:#0A7A75 !important; color:white !important; font-weight:bold !important;
    border-radius:8px !important; width:100% !important;
    text-transform:uppercase !important; border:none !important;
}
.stButton>button:hover { background:#0c9993 !important; }
[data-testid="stMetric"] {
    background:#112233 !important; border-top:3px solid #0A7A75 !important;
    border-radius:8px !important; padding:10px !important;
}
@keyframes blink { 50%{opacity:0;} }
.blink { animation:blink 1s linear infinite; }
@keyframes pg {
    0%  { transform:scale(0.95); box-shadow:0 0 0 0 rgba(26,122,74,.7); }
    70% { transform:scale(1);    box-shadow:0 0 0 10px rgba(26,122,74,0); }
    100%{ transform:scale(0.95); box-shadow:0 0 0 0 rgba(26,122,74,0); }
}
.dot-on  { height:12px; width:12px; background:#1A7A4A; border-radius:50%;
            display:inline-block; animation:pg 2s infinite; margin-right:6px; }
.dot-off { height:12px; width:12px; background:#555; border-radius:50%;
            display:inline-block; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# Show import errors prominently
for err in _import_errors:
    st.sidebar.error(f"❌ {err}")
if _import_errors:
    st.sidebar.warning("Some backend modules failed to load. Stub functions are active.")

# ── Session state defaults ─────────────────────────────────────────────────
_DEFAULTS = dict(
    running=False, mode="live", verdict="CALIBRATING",
    bpm=0.0, confidence=0.0, loop_detected=False,
    signal_quality_str="LOW", is_calibrating=True,
    frame_count=0, session_log=[],
    rppg_signal=[], fft_freqs=[], fft_power=[],
    # raw pipeline values for debug
    raw_sig_std=0.0, raw_sig_quality=0.0, raw_snr=0.0,
    raw_dominant_freq=0.0, raw_threat_score_str="—",
    pulse_present=False, loop_score=0.0,
    camera_index=0, cap=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "face_detector" not in st.session_state:
    st.session_state.face_detector = FaceDetector()

# ── Header ─────────────────────────────────────────────────────────────────
hL, hR = st.columns([8, 2])
with hL:
    st.markdown("""
    <div style='font-family:"Rajdhani",sans-serif;font-size:2.4rem;
                font-weight:bold;color:white;line-height:1;'>
        🛡 rPPG DeepShield
    </div>
    <div style='color:#0A7A75;font-style:italic;font-size:0.9rem;margin-top:2px;'>
        Biometric Liveness Detection System — Remote Photoplethysmography
    </div>""", unsafe_allow_html=True)
with hR:
    if st.session_state.running:
        st.markdown("<div style='text-align:right;padding-top:18px;'>"
                    "<span class='dot-on'></span>"
                    "<span style='color:#1A7A4A;font-weight:bold;'>SYSTEM ACTIVE</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right;padding-top:18px;'>"
                    "<span class='dot-off'></span>"
                    "<span style='color:#555;font-weight:bold;'>STANDBY</span></div>",
                    unsafe_allow_html=True)
st.markdown("<hr style='border-top:2px solid #0A7A75;margin:6px 0 18px 0;'>",
            unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align:center;'>🛡 DeepShield</h2>",
                    unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Parameters")
buf_size = st.sidebar.selectbox("Buffer size (frames)", [150, 300, 450], index=1)
roi_method = st.sidebar.radio("ROI method",
    ["Forehead + Cheeks", "Forehead only", "Cheeks only"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Module Status")
st.sidebar.markdown(
    f"{'✅' if _face_ok   else '❌'} Face Detection  \n"
    f"{'✅' if _engine_ok else '❌'} rPPG Engine  \n"
    f"{'✅' if _threat_ok else '❌'} Threat Analyzer"
)

with st.sidebar.expander("ℹ️ How it works"):
    st.markdown("""
**rPPG** detects the tiny colour changes in skin caused by each heartbeat
(~1–5 digital numbers per channel, 0.7–3.0 Hz band).

**AI videos fail** because generators don't simulate sub-surface blood physics
→ no cardiac frequency → `pulse_present = False` → **THREAT** verdict.

**3 detection layers:**
1. **FFT** — dominant cardiac frequency + SNR check
2. **Autocorrelation** — synthetic loop detection
3. **Scoring** — weighted evidence combination
""")

# ── Two-column layout ──────────────────────────────────────────────────────
col_left, col_right = st.columns([4, 6])

# ── LEFT: controls + verdict + metrics ────────────────────────────────────
with col_left:
    # Mode selector
    st.markdown("<p style='font-size:.85rem;color:#9ca3af;margin-bottom:4px;'>INPUT MODE</p>",
                unsafe_allow_html=True)
    mode_sel = st.radio("mode", ["📷  Live Camera", "🎬  Upload Video"],
                         horizontal=True, label_visibility="collapsed")
    st.session_state.mode = "live" if "Camera" in mode_sel else "upload"

    def _reset_all():
        """Flush every stateful singleton before a new session."""
        reset_engine()
        reset_verdict_history()
        st.session_state.frame_count         = 0
        st.session_state.rppg_signal         = []
        st.session_state.fft_freqs           = []
        st.session_state.fft_power           = []
        st.session_state.verdict             = "CALIBRATING"
        st.session_state.bpm                 = 0.0
        st.session_state.confidence          = 0.0
        st.session_state.is_calibrating      = True
        st.session_state.session_log         = []
        st.session_state.pulse_present       = False
        st.session_state.raw_sig_std         = 0.0
        st.session_state.raw_sig_quality     = 0.0
        st.session_state.raw_snr             = 0.0
        st.session_state.raw_dominant_freq   = 0.0
        st.session_state.face_detector.reset()

    if st.session_state.mode == "live":
        cam_opt = st.selectbox("Camera", ["Camera 0 (Default)", "Camera 1", "Camera 2"])
        st.session_state.camera_index = 0 if "0" in cam_opt else int(cam_opt.split()[1])
        if not st.session_state.running:
            if st.button("▶  START ANALYSIS", key="btn_start_live"):
                _reset_all()
                st.session_state.running = True
                st.rerun()
        else:
            if st.button("⏹  STOP", key="btn_stop_live"):
                st.session_state.running = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.rerun()
    else:
        uploaded = st.file_uploader("Drop video here", type=["mp4","avi","mov"])
        if uploaded:
            st.success(f"✅ {uploaded.name}")
            if not st.session_state.running:
                if st.button("▶  ANALYSE VIDEO", key="btn_start_upload"):
                    tmp = os.path.join(PROJECT_ROOT, "temp_upload.mp4")
                    with open(tmp, "wb") as fh:
                        fh.write(uploaded.getbuffer())
                    _reset_all()
                    st.session_state.running = True
                    st.rerun()
            else:
                if st.button("⏹  STOP", key="btn_stop_upload"):
                    st.session_state.running = False
                    if st.session_state.cap:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Placeholders for dynamic left-column content ──────────────────────
    calib_ph   = st.empty()
    verdict_ph = st.empty()
    metrics_ph = st.empty()

    st.markdown("---")
    st.markdown("#### 📋 Session Log")
    log_ph = st.empty()

# ── RIGHT: video + charts + debug ─────────────────────────────────────────
with col_right:
    vid_title = "🎥 LIVE FEED" if st.session_state.mode == "live" else "🎬 VIDEO ANALYSIS"
    st.markdown(f"#### {vid_title}")
    frame_ph = st.empty()

    mc1, mc2, mc3 = st.columns(3)
    res_ph = mc1.empty()
    fps_ph = mc2.empty()
    fc_ph  = mc3.empty()

    st.markdown("#### 📈 Cardiac Signal Waveform")
    chart_ph = st.empty()        # rPPG waveform lives here

    st.markdown("#### 📊 Frequency Spectrum")
    fft_ph = st.empty()          # FFT bar chart lives here

    st.markdown("#### 🔧 Live Pipeline Debug")
    debug_ph = st.empty()        # raw values live here


# ── render_ui_state — pure placeholder updater ────────────────────────────
def render_ui_state():
    v   = st.session_state.verdict
    cal = st.session_state.is_calibrating

    # 1. Calibration bar
    if st.session_state.running and cal:
        with calib_ph.container():
            prog = min(1.0, st.session_state.frame_count / 150.0)
            st.markdown(f"<div style='color:#C9A84C;font-weight:bold;'>"
                        f"⏳ CALIBRATING — {int(prog*100)}%</div>",
                        unsafe_allow_html=True)
            st.progress(prog)
    else:
        calib_ph.empty()

    # 2. Verdict card
    if v == "REAL":
        card = ("<div style='background:linear-gradient(135deg,#0D3B1F,#1A7A4A);"
                "border:2px solid #1A7A4A;border-radius:12px;padding:20px;"
                "text-align:center;min-height:150px;"
                "box-shadow:0 0 18px rgba(26,122,74,.5);'>"
                "<div style='font-size:2.8rem;'>✅</div>"
                "<div style='font-size:1.4rem;font-weight:bold;color:white;'>"
                "REAL HUMAN DETECTED</div>"
                "<div style='color:#a7f3d0;font-size:.85rem;'>Biological pulse confirmed</div>"
                "</div>")
    elif v == "THREAT":
        card = ("<div style='background:linear-gradient(135deg,#3B0D0D,#C0392B);"
                "border:2px solid #C0392B;border-radius:12px;padding:20px;"
                "text-align:center;min-height:150px;"
                "box-shadow:0 0 18px rgba(192,57,43,.5);'>"
                "<div style='font-size:2.8rem;' class='blink'>⚠️</div>"
                "<div style='font-size:1.4rem;font-weight:bold;color:white;'>"
                "DEEPFAKE DETECTED</div>"
                "<div style='color:#fca5a5;font-size:.85rem;'>"
                "No biological signal — CYBER THREAT</div>"
                "</div>")
    elif v == "UNCERTAIN":
        card = ("<div style='background:#1A1A0D;border:2px solid #C9A84C;"
                "border-radius:12px;padding:20px;text-align:center;min-height:150px;'>"
                "<div style='font-size:2.8rem;'>🔍</div>"
                "<div style='font-size:1.4rem;font-weight:bold;color:#C9A84C;'>"
                "ANALYSING…</div>"
                "<div style='color:#D5D8DC;font-size:.85rem;'>"
                "Signal insufficient — keep face still</div>"
                "</div>")
    else:
        card = ("<div style='background:#111827;border:2px solid #374151;"
                "border-radius:12px;padding:20px;text-align:center;min-height:150px;'>"
                "<div style='font-size:2.8rem;'>⏳</div>"
                "<div style='font-size:1.4rem;font-weight:bold;color:#9ca3af;'>"
                "CALIBRATING</div>"
                "<div style='color:#6b7280;font-size:.85rem;'>"
                "Building signal buffer…</div>"
                "</div>")
    verdict_ph.markdown(card, unsafe_allow_html=True)

    # 3. Metrics (post-calibration only)
    if not cal:
        conf_pct = st.session_state.confidence * 100
        with metrics_ph.container():
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            c1.metric("🫀 Heart Rate",    f"{st.session_state.bpm:.0f} BPM")
            c2.metric("📊 Confidence",    f"{conf_pct:.0f}% ({v})")
            c3.metric("🔁 Loop Detected", "YES ⚠️" if st.session_state.loop_detected else "NO ✓")
            c4.metric("⚡ Sig Quality",   st.session_state.signal_quality_str)
    else:
        metrics_ph.empty()

    # 4. Waveform chart  — key="rppg_waveform" prevents duplicate-ID error
    sig = st.session_state.rppg_signal
    if len(sig) >= 10:
        line_col = ("#1A7A4A" if v == "REAL"
                    else "#C0392B" if v == "THREAT"
                    else "#C9A84C")
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            y=sig[-150:], mode="lines",
            line=dict(color=line_col, width=1.8),
            name="rPPG",
        ))
        fig_w.add_hline(y=0, line_dash="dot", line_color="#555")
        fig_w.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            margin=dict(l=0, r=0, t=6, b=0), height=180,
            xaxis=dict(title="Frames (last 150)", showgrid=False,
                       tickfont=dict(color="#aaa")),
            yaxis=dict(title="Amplitude", showgrid=True,
                       gridcolor="#1e3a5f", tickfont=dict(color="#aaa")),
            showlegend=False,
        )
        chart_ph.plotly_chart(fig_w, use_container_width=True,
                               config={"displayModeBar": False},
                               key="rppg_waveform")
    else:
        chart_ph.markdown("<div style='color:#555;text-align:center;padding:30px;'>"
                          "Waiting for signal…</div>", unsafe_allow_html=True)

    # 5. FFT spectrum chart  — key="fft_spectrum" prevents duplicate-ID error
    freqs = st.session_state.fft_freqs
    power = st.session_state.fft_power
    if len(freqs) > 4:
        fr = np.array(freqs)
        pw = np.array(power)
        # Mask to 0–4 Hz for display
        mask = fr <= 4.0
        fr, pw = fr[mask], pw[mask]
        bar_cols = ["#0A7A75"] * len(fr)
        if len(pw) > 0:
            peak_i = int(np.argmax(pw))
            bar_cols[peak_i] = "#FFFFFF"
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(x=fr, y=pw, marker_color=bar_cols))
        fig_f.add_vline(x=0.7, line_dash="dash", line_color="#C9A84C",
                        annotation_text="Low HR", annotation_font_color="#C9A84C")
        fig_f.add_vline(x=3.0, line_dash="dash", line_color="#C9A84C",
                        annotation_text="High HR", annotation_font_color="#C9A84C")
        if len(pw) > 0:
            fig_f.add_annotation(
                x=fr[peak_i], y=pw[peak_i],
                text=f"{fr[peak_i]*60:.0f} BPM",
                showarrow=True, arrowhead=2,
                font=dict(color="white", size=11),
            )
        fig_f.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            margin=dict(l=0, r=0, t=6, b=0), height=200,
            xaxis=dict(title="Frequency (Hz)", showgrid=False,
                       tickfont=dict(color="#aaa")),
            yaxis=dict(title="Power", showgrid=False,
                       tickfont=dict(color="#aaa")),
            showlegend=False,
        )
        fft_ph.plotly_chart(fig_f, use_container_width=True,
                             config={"displayModeBar": False},
                             key="fft_spectrum")
    else:
        fft_ph.markdown("<div style='color:#555;text-align:center;padding:20px;'>"
                        "Waiting for FFT data…</div>", unsafe_allow_html=True)

    # 6. Live debug table — raw pipeline values every frame
    pp   = st.session_state.pulse_present
    snr  = st.session_state.raw_snr
    df   = st.session_state.raw_dominant_freq
    std  = st.session_state.raw_sig_std
    sq   = st.session_state.raw_sig_quality
    loop = st.session_state.loop_detected
    ls   = st.session_state.loop_score
    debug_ph.markdown(f"""
<div style='background:#0a1120;border:1px solid #1e3a5f;border-radius:8px;
            padding:10px 14px;font-family:monospace;font-size:.78rem;color:#aad;'>
<b style='color:#0A7A75;'>PIPELINE VALUES (live)</b><br>
Verdict        : <b style='color:{"#22c55e" if v=="REAL" else "#ef4444" if v=="THREAT" else "#eab308"}'>{v}</b><br>
Confidence     : {st.session_state.confidence*100:.1f}%<br>
Pulse present  : {"✅ YES" if pp else "❌ NO"}<br>
BPM            : {st.session_state.bpm:.1f}<br>
Dominant freq  : {df:.3f} Hz<br>
SNR score      : {snr:.3f}  (threshold 5.0)<br>
Loop detected  : {"YES" if loop else "no"}  (score={ls:.3f})<br>
Signal std     : {std:.6f}  (0 = flat = AI face)<br>
Signal quality : {sq:.4f}<br>
Sig quality str: {st.session_state.signal_quality_str}<br>
Is calibrating : {cal}<br>
Frame count    : {st.session_state.frame_count}<br>
</div>""", unsafe_allow_html=True)

    # 7. Session log
    if st.session_state.session_log:
        rows = st.session_state.session_log[-20:]
        html = ("<div style='max-height:180px;overflow-y:auto;background:#0a1120;"
                "padding:8px 12px;border-radius:8px;font-family:monospace;"
                "font-size:.78rem;border:1px solid #1e3a5f;'>")
        for e in reversed(rows):
            c = "#22c55e" if "REAL" in e else "#ef4444" if "THREAT" in e else "#eab308"
            html += f"<div style='color:{c};margin-bottom:2px;'>{e}</div>"
        html += "</div>"
        log_ph.markdown(html, unsafe_allow_html=True)
    else:
        log_ph.markdown("<div style='color:#555;font-size:.8rem;font-style:italic;'>"
                        "No log entries yet…</div>", unsafe_allow_html=True)


# ── MAIN LOOP ──────────────────────────────────────────────────────────────
if not st.session_state.running:
    render_ui_state()
    st.stop()

# ── Open capture ───────────────────────────────────────────────────────────
try:
    if st.session_state.cap is None:
        if st.session_state.mode == "live":
            st.session_state.cap = cv2.VideoCapture(st.session_state.camera_index)
        else:
            tmp = os.path.join(PROJECT_ROOT, "temp_upload.mp4")
            st.session_state.cap = cv2.VideoCapture(tmp)

    cap = st.session_state.cap
    if not cap or not cap.isOpened():
        st.error("❌ Cannot open video source. Check camera index or uploaded file.")
        st.session_state.running = False
        st.session_state.cap = None
        st.stop()

    # ── Frame loop ─────────────────────────────────────────────────────────
    while st.session_state.running:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.release()
            st.session_state.cap = None
            st.session_state.running = False
            st.info("✅ Video ended.")
            render_ui_state()
            break

        # ── 1. Face Detection ─────────────────────────────────────────────
        try:
            face_detected, face_bbox = st.session_state.face_detector.detect(frame)
        except Exception:
            face_detected, face_bbox = False, (0, 0, 0, 0)

        if not face_detected:
            overlay = frame.copy()
            cv2.putText(overlay, "NO FACE DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2)
            frame_ph.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)
            st.session_state.verdict = "UNCERTAIN"
            render_ui_state()
            time.sleep(0.033)
            continue

        x, y, w, h = face_bbox

        # ── 2. ROI Extraction ─────────────────────────────────────────────
        try:
            roi_f, roi_c = extract_roi(frame, face_bbox)
            if roi_method == "Forehead only":
                roi_c = roi_f
            elif roi_method == "Cheeks only":
                roi_f = roi_c
            pkt = create_data_packet(frame, face_detected, face_bbox, roi_f, roi_c)
        except Exception:
            roi_f = roi_c = np.zeros((10, 10, 3), np.uint8)
            pkt = {"roi_forehead": roi_f, "roi_cheeks": roi_c}

        # ── 3. rPPG Engine ────────────────────────────────────────────────
        try:
            eng = process_signal(
                roi_forehead=pkt["roi_forehead"],
                roi_cheeks=pkt["roi_cheeks"],
                frame_count=st.session_state.frame_count,
                buffer_size=buf_size,
            )
            rppg_sig   = np.array(eng.get("rppg_signal", []), dtype=np.float64)
            est_bpm    = float(eng.get("estimated_bpm", 0.0))
            is_cal     = bool(eng.get("is_calibrating", True))
            sig_qual   = float(eng.get("signal_quality", 0.0))
        except Exception as exc:
            rppg_sig = np.zeros(10)
            est_bpm, is_cal, sig_qual = 0.0, True, 0.0

        # ── 4. Threat Analysis ────────────────────────────────────────────
        try:
            result = analyze_threat(rppg_sig, est_bpm, is_cal, sig_qual)
        except Exception:
            result = {"verdict": "UNCERTAIN", "bpm": 0.0, "confidence": 0.0,
                      "pulse_present": False, "loop_detected": False,
                      "snr_score": 0.0, "dominant_freq_hz": 0.0,
                      "loop_score": 0.0}

        # ── 5. Push results into session state ────────────────────────────
        v = result.get("verdict", "UNCERTAIN") if not is_cal else "CALIBRATING"
        st.session_state.verdict             = v
        st.session_state.is_calibrating      = is_cal
        st.session_state.bpm                 = result.get("bpm", 0.0)
        st.session_state.confidence          = result.get("confidence", 0.0)
        st.session_state.loop_detected       = result.get("loop_detected", False)
        st.session_state.loop_score          = result.get("loop_score", 0.0)
        st.session_state.pulse_present       = result.get("pulse_present", False)
        # raw debug values
        st.session_state.raw_sig_std         = float(np.std(rppg_sig))
        st.session_state.raw_sig_quality     = sig_qual
        st.session_state.raw_snr             = result.get("snr_score", 0.0)
        st.session_state.raw_dominant_freq   = result.get("dominant_freq_hz", 0.0)
        sq_str = "HIGH" if sig_qual > 0.7 else "MEDIUM" if sig_qual > 0.4 else "LOW"
        st.session_state.signal_quality_str  = sq_str
        st.session_state.rppg_signal         = rppg_sig.tolist()

        # FFT for spectrum display
        if len(rppg_sig) > 10:
            fft_out = np.abs(np.fft.rfft(rppg_sig))
            freqs_arr = np.fft.rfftfreq(len(rppg_sig), 1.0 / 30.0)
            st.session_state.fft_freqs = freqs_arr.tolist()
            st.session_state.fft_power = fft_out.tolist()

        # ── 6. Draw overlays ──────────────────────────────────────────────
        box_col = ((0,255,0) if v == "REAL"
                   else (0,0,255) if v == "THREAT"
                   else (0,215,255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_col, 3)
        # Forehead ROI indicator
        fx0 = x + int(w*0.2); fx1 = x + int(w*0.8); fy1 = y + int(h*0.3)
        cv2.rectangle(frame, (fx0, y), (fx1, fy1), (10, 122, 117), 1)
        cv2.putText(frame, v, (x, max(14, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, box_col, 2)
        H, W = frame.shape[:2]
        cv2.rectangle(frame, (8, H-46), (220, H-8), (0,0,0), -1)
        cv2.putText(frame, f"BPM: {st.session_state.bpm:.0f}",
                    (12, H-22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        curr_fps = 1.0 / max(time.time() - t0, 1e-4)
        cv2.putText(frame, f"FPS:{curr_fps:.0f}",
                    (W-90, H-22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_container_width=True)
        res_ph.metric("Resolution", f"{W}×{H}")
        fps_ph.metric("FPS",        f"{curr_fps:.0f}")
        fc_ph.metric("Frames",      f"{st.session_state.frame_count}")

        # Session log entry every 30 frames after calibration
        fc = st.session_state.frame_count
        if not is_cal and fc > 0 and fc % 30 == 0:
            ts = datetime.datetime.now().strftime("%M:%S")
            entry = (f"[{ts}] {v:<9}"
                     f"  BPM={st.session_state.bpm:3.0f}"
                     f"  conf={st.session_state.confidence*100:3.0f}%"
                     f"  SNR={st.session_state.raw_snr:.1f}"
                     f"  pulse={'Y' if st.session_state.pulse_present else 'N'}")
            st.session_state.session_log.append(entry)

        render_ui_state()
        st.session_state.frame_count += 1
        time.sleep(0.033)

except Exception as exc:
    import traceback
    st.error(f"❌ Pipeline error: {exc}")
    st.code(traceback.format_exc())
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
