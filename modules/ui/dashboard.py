import streamlit as st
import cv2
import numpy as np
import scipy
import plotly.graph_objects as go
import time
import threading
import io
import csv
import datetime
import sys
import os

# ── Project root on path ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="rPPG DeepShield",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Module imports with graceful stubs ────────────────────────────────────
_import_errors = []

try:
    from modules.face_detection import FaceDetector, extract_roi, create_data_packet
except Exception as e:
    _import_errors.append(f"FaceDetector: {e}")
    class FaceDetector:
        def detect(self, frame): return False, (0, 0, 0, 0)
        def reset(self): pass
    def extract_roi(frame, bbox):
        return np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)
    def create_data_packet(*args, **kwargs):
        return {"roi_forehead": np.zeros((10, 10, 3), dtype=np.uint8),
                "roi_cheeks":   np.zeros((10, 10, 3), dtype=np.uint8)}

try:
    from modules.rppg_engine import process_signal, reset_engine
except Exception as e:
    _import_errors.append(f"rPPG Engine: {e}")
    def process_signal(*args, **kwargs):
        fc = kwargs.get("frame_count", 0)
        buf = kwargs.get("buffer_size", 300)
        return {"rppg_signal": np.zeros(buf), "estimated_bpm": 0.0,
                "is_calibrating": fc < 150, "signal_quality": 0.0}
    def reset_engine(): pass

try:
    from modules.threat_analyzer import analyze_threat
    from modules.threat_analyzer.threat_scorer import reset_verdict_history
except Exception as e:
    _import_errors.append(f"ThreatAnalyzer: {e}")
    def analyze_threat(*args, **kwargs):
        return {"verdict": "UNCERTAIN", "bpm": 0.0, "confidence": 0.0,
                "pulse_present": False, "loop_detected": False,
                "snr_score": 0.0, "dominant_freq_hz": 0.0}
    def reset_verdict_history(): pass

# ── CSS Theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');

body, .stApp { background-color: #0D1B2A !important; }
h1,h2,h3,h4,h5,h6 { font-family:'Rajdhani',sans-serif !important; color:white !important; }
p,div,span,label { color:#D5D8DC !important; }

[data-testid="stSidebar"] { background-color:#0A1628 !important; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

.stButton>button {
    background-color:#0A7A75 !important; color:white !important;
    font-weight:bold !important; border-radius:8px !important;
    width:100% !important; text-transform:uppercase !important; border:none !important;
}
.stButton>button:hover { background-color:#0c9993 !important; }

[data-testid="stMetric"] {
    background-color:#112233 !important; border-top:3px solid #0A7A75 !important;
    border-radius:8px !important; padding:10px !important;
    box-shadow:0px 4px 6px rgba(0,0,0,0.3);
}
.streamlit-expanderHeader { background-color:#112233 !important; color:white !important; }

@keyframes blink { 50%{opacity:0;} }
.blink { animation:blink 1s linear infinite; }

@keyframes pulse-green {
    0%{transform:scale(0.95);box-shadow:0 0 0 0 rgba(26,122,74,0.7);}
    70%{transform:scale(1);box-shadow:0 0 0 10px rgba(26,122,74,0);}
    100%{transform:scale(0.95);box-shadow:0 0 0 0 rgba(26,122,74,0);}
}
.dot-active{height:12px;width:12px;background:#1A7A4A;border-radius:50%;
    display:inline-block;animation:pulse-green 2s infinite;margin-right:8px;}
.dot-standby{height:12px;width:12px;background:#555;border-radius:50%;
    display:inline-block;margin-right:8px;}
</style>
""", unsafe_allow_html=True)

# ── Show import warnings inline ───────────────────────────────────────────
for err in _import_errors:
    st.sidebar.warning(f"⚠️ {err}")

# ── Session state init ────────────────────────────────────────────────────
_defaults = {
    "running": False, "mode": "live", "verdict": "CALIBRATING",
    "bpm": 0.0, "confidence": 0.0, "loop_detected": False,
    "signal_quality": "LOW", "rppg_signal": [], "fft_freqs": [],
    "fft_power": [], "frame_count": 0, "session_log": [],
    "is_calibrating": True, "dominant_freq": 0.0, "snr_score": 0.0,
    "autocorr_score": 0.0, "cap": None, "camera_index": 0,
    "pulse_present": False,
    # live debug values
    "dbg_sig_std": 0.0, "dbg_sig_quality_raw": 0.0,
    "dbg_snr_raw": 0.0, "dbg_dom_freq": 0.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# face_detector is a class instance — only init once
if "face_detector" not in st.session_state:
    st.session_state.face_detector = FaceDetector()

# ── Header ────────────────────────────────────────────────────────────────
c_title, c_status = st.columns([8, 2])
with c_title:
    st.markdown("""
    <div style='font-family:"Rajdhani",sans-serif;font-size:2.4rem;font-weight:bold;color:white;'>
        🛡 rPPG DeepShield
    </div>
    <div style='color:#0A7A75;font-style:italic;font-size:0.95rem;margin-top:-6px;'>
        Biometric Liveness Detection System — Powered by Remote Photoplethysmography
    </div>
    """, unsafe_allow_html=True)
with c_status:
    if st.session_state.running:
        st.markdown("<div style='text-align:right;margin-top:20px;'><span class='dot-active'></span>"
                    "<span style='color:#1A7A4A;font-weight:bold;'>SYSTEM ACTIVE</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right;margin-top:20px;'><span class='dot-standby'></span>"
                    "<span style='color:#777;font-weight:bold;'>STANDBY</span></div>",
                    unsafe_allow_html=True)
st.markdown("<hr style='border-top:2px solid #0A7A75;margin:6px 0 20px 0;'>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align:center;'>🛡 DeepShield</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Detection Parameters")
bp_low      = st.sidebar.slider("Bandpass Low (Hz)",   0.5, 1.0,  0.7,  step=0.05)
bp_high     = st.sidebar.slider("Bandpass High (Hz)",  2.0, 4.0,  3.0,  step=0.1)
threat_thr  = st.sidebar.slider("Threat Threshold",    0.50, 0.99, 0.75, step=0.01)
loop_sens   = st.sidebar.slider("Loop Sensitivity",    0.80, 0.99, 0.92, step=0.01)
buf_size_val= st.sidebar.selectbox("Buffer Size (frames)", [150, 300, 450], index=1)
roi_method  = st.sidebar.radio("ROI Method",
                ["Forehead + Cheeks", "Forehead Only", "Cheeks Only"])

with st.sidebar.expander("ℹ️ About rPPG"):
    st.markdown("""
**Remote Photoplethysmography (rPPG)** detects microscopic colour changes in
skin caused by the heartbeat. Each pump causes a subtle flush of blood beneath
the skin surface, shifting the R/G/B channels by ≈1–5 digital numbers.

**Why deepfakes fail:** AI video generators do not simulate sub-surface blood
flow physics. The rendered skin has *no* cardiac frequency in the 0.7–3.0 Hz
band — no pulse, no signal.

**The 3 detection layers:**
1. **FFT** — finds dominant frequency; must be in the cardiac band.
2. **Autocorrelation** — checks for synthetic loop repetition.
3. **SNR** — biological signals have a characteristic signal-to-noise ratio.

*de Haan & Jeanne 2013; Wang et al. 2017 (CHROM method)*
    """)

with st.sidebar.expander("🧪 Test Signals"):
    if st.sidebar.button("▶ Simulate REAL Signal"):
        st.session_state.running = False
        t = np.linspace(0, 10, buf_size_val)
        st.session_state.rppg_signal = list(
            np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.15, buf_size_val))
        st.session_state.verdict        = "REAL"
        st.session_state.bpm            = 72.0
        st.session_state.confidence     = 0.92
        st.session_state.dominant_freq  = 1.2
        st.session_state.snr_score      = 8.4
        st.session_state.autocorr_score = 0.41
        st.session_state.signal_quality = "HIGH"
        st.session_state.pulse_present  = True
        st.session_state.is_calibrating = False
    if st.sidebar.button("▶ Simulate DEEPFAKE Signal"):
        st.session_state.running = False
        t = np.linspace(0, 10, buf_size_val)
        st.session_state.rppg_signal = list(np.zeros(buf_size_val))
        st.session_state.verdict        = "THREAT"
        st.session_state.bpm            = 0.0
        st.session_state.confidence     = 0.94
        st.session_state.dominant_freq  = 0.0
        st.session_state.snr_score      = 0.0
        st.session_state.autocorr_score = 0.0
        st.session_state.signal_quality = "LOW"
        st.session_state.pulse_present  = False
        st.session_state.is_calibrating = False

# ── Main layout ───────────────────────────────────────────────────────────
col_ctrl, col_vid = st.columns([4, 6])

# ── Left column: controls ─────────────────────────────────────────────────
with col_ctrl:
    st.markdown("<p style='font-size:0.85rem;margin-bottom:4px;color:#9ca3af;'>SELECT INPUT MODE</p>",
                unsafe_allow_html=True)
    mode_sel = st.radio("mode", ["📷  Live Camera Feed", "🎬  Upload Demo Video"],
                         horizontal=True, label_visibility="collapsed")
    st.session_state.mode = "live" if "Camera" in mode_sel else "upload"

    def _do_start_reset():
        """Reset all stateful module singletons before every new session."""
        reset_engine()
        reset_verdict_history()
        st.session_state.frame_count    = 0
        st.session_state.rppg_signal    = []
        st.session_state.fft_freqs      = []
        st.session_state.fft_power      = []
        st.session_state.verdict        = "CALIBRATING"
        st.session_state.bpm            = 0.0
        st.session_state.confidence     = 0.0
        st.session_state.is_calibrating = True
        st.session_state.session_log    = []
        st.session_state.pulse_present  = False
        st.session_state.face_detector.reset()
        st.session_state.running        = True

    if st.session_state.mode == "live":
        cam_opt = st.selectbox("Camera Device",
                               ["Camera 0 (Default)", "Camera 1", "Camera 2"])
        st.session_state.camera_index = (
            0 if cam_opt == "Camera 0 (Default)"
            else int(cam_opt.split()[1])
        )
        if not st.session_state.running:
            if st.button("▶  START ANALYSIS"):
                _do_start_reset()
                st.rerun()
        else:
            st.markdown("""<style>
            div[data-testid="stHorizontalBlock"] div.stButton>button
            {background-color:#C0392B!important;}
            </style>""", unsafe_allow_html=True)
            if st.button("⏹  STOP ANALYSIS"):
                st.session_state.running = False
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.rerun()
    else:
        uploaded = st.file_uploader("Drop video file here", type=["mp4", "avi", "mov"])
        if uploaded:
            st.markdown(f"<div style='background:#1A7A4A;color:white;padding:4px 10px;"
                        f"border-radius:4px;font-size:12px;display:inline-block;"
                        f"margin-bottom:8px;'>✅ {uploaded.name}</div>", unsafe_allow_html=True)
            if not st.session_state.running:
                if st.button("▶  ANALYZE VIDEO"):
                    tmp = os.path.join(PROJECT_ROOT, "temp_upload.mp4")
                    with open(tmp, "wb") as f:
                        f.write(uploaded.getbuffer())
                    _do_start_reset()
                    st.rerun()
            else:
                if st.button("⏹  STOP ANALYSIS"):
                    st.session_state.running = False
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Placeholders (filled by render_ui_state)
    calib_ph    = st.empty()
    verdict_ph  = st.empty()
    metrics_ph  = st.empty()
    breakdown_ph= st.empty()

    st.markdown("### 📋 Session Log")
    log_ph = st.empty()

    if st.session_state.session_log:
        csv_rows = "Timestamp,Verdict,BPM,Confidence\n" + "\n".join(st.session_state.session_log)
        st.download_button("⬇ Export CSV", csv_rows, "session_log.csv", "text/csv")

# ── Right column: video + charts ──────────────────────────────────────────
with col_vid:
    vid_label = "🎥 LIVE FEED" if st.session_state.mode == "live" else "🎬 VIDEO ANALYSIS"
    st.markdown(f"### {vid_label}")
    frame_ph = st.empty()

    mc1, mc2, mc3 = st.columns(3)
    res_ph = mc1.empty()
    fps_ph = mc2.empty()
    fc_ph  = mc3.empty()

    st.markdown("### 📈 Cardiac Signal Waveform")
    chart_ph = st.empty()
    fft_ph   = st.empty()

    # ── Live Debug Panel ─────────────────────────────────────────────────
    with st.expander("🔧 Pipeline Debug (live values)", expanded=False):
        debug_ph = st.empty()


# ── UI Renderer ───────────────────────────────────────────────────────────
def render_ui_state():
    v   = st.session_state.verdict
    cal = st.session_state.is_calibrating

    # Calibration bar
    if st.session_state.running and cal:
        with calib_ph.container():
            st.markdown("<div style='color:#C9A84C;font-weight:bold;'>⏳ CALIBRATING SIGNAL BUFFER</div>",
                        unsafe_allow_html=True)
            st.progress(min(1.0, st.session_state.frame_count / 150.0))
            st.markdown("<div style='color:#777;font-size:0.8rem;font-style:italic;'>"
                        "Collecting 150 frames before analysis begins...</div>", unsafe_allow_html=True)
    else:
        calib_ph.empty()

    # Verdict card
    if v == "REAL":
        card = ("<div style='background:linear-gradient(135deg,#0D3B1F,#1A7A4A);"
                "border:2px solid #1A7A4A;border-radius:12px;padding:24px;text-align:center;"
                "min-height:160px;box-shadow:0 0 18px rgba(26,122,74,0.5);'>"
                "<div style='font-size:3rem;'>✅</div>"
                "<div style='font-size:1.5rem;font-weight:bold;color:white;'>REAL HUMAN DETECTED</div>"
                "<div style='color:#a7f3d0;font-size:0.9rem;'>Biological pulse confirmed</div></div>")
    elif v == "THREAT":
        card = ("<div style='background:linear-gradient(135deg,#3B0D0D,#C0392B);"
                "border:2px solid #C0392B;border-radius:12px;padding:24px;text-align:center;"
                "min-height:160px;box-shadow:0 0 18px rgba(192,57,43,0.5);'>"
                "<div style='font-size:3rem;' class='blink'>⚠️</div>"
                "<div style='font-size:1.5rem;font-weight:bold;color:white;'>DEEPFAKE DETECTED</div>"
                "<div style='color:#fca5a5;font-size:0.9rem;'>No biological signal — CYBER THREAT</div></div>")
    elif v == "UNCERTAIN":
        card = ("<div style='background:#1A1A0D;border:2px solid #C9A84C;"
                "border-radius:12px;padding:24px;text-align:center;min-height:160px;'>"
                "<div style='font-size:3rem;'>🔍</div>"
                "<div style='font-size:1.5rem;font-weight:bold;color:#C9A84C;'>ANALYZING...</div>"
                "<div style='color:#D5D8DC;font-size:0.9rem;'>Insufficient signal — keep face still</div></div>")
    else:
        card = ("<div style='background:#111827;border:2px solid #374151;"
                "border-radius:12px;padding:24px;text-align:center;min-height:160px;'>"
                "<div style='font-size:3rem;'>⏳</div>"
                "<div style='font-size:1.5rem;font-weight:bold;color:#9ca3af;'>CALIBRATING</div>"
                "<div style='color:#6b7280;font-size:0.9rem;'>Building signal buffer — please wait</div></div>")
    verdict_ph.markdown(card, unsafe_allow_html=True)

    # Metrics (only post-calibration)
    if not cal:
        with metrics_ph.container():
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            bpm_val  = st.session_state.bpm
            conf_pct = st.session_state.confidence * 100
            # Contextualise confidence label by verdict
            conf_label = (f"{conf_pct:.0f}% {v}" if v in ("REAL","THREAT","UNCERTAIN")
                          else f"{conf_pct:.0f}%")
            c1.metric("🫀 HEART RATE",    f"{bpm_val:.0f} BPM")
            c2.metric("📊 CONFIDENCE",    conf_label)
            loop_str = "YES ⚠️" if st.session_state.loop_detected else "NO ✓"
            c3.metric("🔁 LOOP DETECTED", loop_str)
            c4.metric("⚡ SIGNAL QUALITY", st.session_state.signal_quality)
    else:
        metrics_ph.empty()

    # Detection details expander
    if not cal:
        with breakdown_ph.container():
            with st.expander("🔬 Detection Details", expanded=False):
                dfreq = st.session_state.dominant_freq
                snr   = st.session_state.snr_score
                ac    = st.session_state.autocorr_score
                pp    = st.session_state.pulse_present

                st.markdown(f"**Pulse Present:** {'✅ YES' if pp else '❌ NO'}")
                st.markdown(f"**FFT Dominant Freq:** {dfreq:.3f} Hz → {dfreq*60:.1f} BPM")
                st.progress(min(1.0, dfreq / 3.0))
                st.markdown(f"**SNR Score:** {snr:.2f}  *(threshold 5.0 — higher = more biological)*")
                st.progress(min(1.0, snr / 15.0))
                st.markdown(f"**Autocorr Score:** {ac:.3f}  *(>0.92 = synthetic loop)*")
                st.progress(min(1.0, ac))
    else:
        breakdown_ph.empty()

    # Session log
    if st.session_state.session_log:
        rows = st.session_state.session_log[-20:]
        html = ("<div style='max-height:180px;overflow-y:auto;background:#111827;"
                "padding:10px;border-radius:8px;font-family:monospace;border:1px solid #1f2937;'>")
        for e in reversed(rows):
            col = "#22c55e" if "REAL" in e else "#ef4444" if "THREAT" in e else "#eab308"
            html += f"<div style='color:{col};margin-bottom:3px;'>{e}</div>"
        html += "</div>"
        log_ph.markdown(html, unsafe_allow_html=True)
    else:
        log_ph.markdown("<div style='color:#555;font-size:0.8rem;font-style:italic;'>"
                        "No entries yet...</div>", unsafe_allow_html=True)

    # Signal waveform
    sig = st.session_state.rppg_signal
    if len(sig) >= 2:
        line_col = ("#0A7A75" if v in ("REAL", "CALIBRATING")
                    else "#C0392B" if v == "THREAT" else "#C9A84C")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sig[-150:], mode="lines",
                                  line=dict(color=line_col, width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="#ef4444",
                      annotation_text="Baseline")
        fig.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            margin=dict(l=0, r=0, t=10, b=0), height=200,
            xaxis=dict(title="Frames", showgrid=False,
                       tickfont=dict(color="white")),
            yaxis=dict(title="Amplitude (norm.)", showgrid=True,
                       gridcolor="#1e3a5f", tickfont=dict(color="white")),
            showlegend=False,
        )
        chart_ph.plotly_chart(fig, use_container_width=True,
                               config={"displayModeBar": False})
    else:
        chart_ph.empty()

    # FFT spectrum
    if len(st.session_state.fft_freqs) > 2:
        with fft_ph.container():
            with st.expander("📊 Frequency Spectrum", expanded=False):
                pw   = np.array(st.session_state.fft_power)
                fr   = np.array(st.session_state.fft_freqs)
                cols = ["#0A7A75"] * len(fr)
                if len(pw) > 0:
                    mi = int(np.argmax(pw))
                    cols[mi] = "#FFFFFF"
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=fr, y=pw, marker_color=cols))
                fig2.add_vline(x=0.7, line_dash="dash", line_color="#C9A84C",
                               annotation_text="Min HR")
                fig2.add_vline(x=3.0, line_dash="dash", line_color="#C9A84C",
                               annotation_text="Max HR")
                if len(pw) > 0:
                    fig2.add_annotation(x=fr[mi], y=pw[mi],
                                        text=f"{fr[mi]*60:.0f} BPM",
                                        showarrow=True, arrowhead=1,
                                        font=dict(color="white"))
                fig2.update_layout(
                    plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
                    margin=dict(l=0, r=0, t=10, b=0), height=220,
                    xaxis=dict(title="Frequency (Hz)", range=[0, 4],
                               showgrid=False, tickfont=dict(color="white")),
                    yaxis=dict(title="Power", showgrid=False,
                               tickfont=dict(color="white")),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True,
                                 config={"displayModeBar": False})
    else:
        fft_ph.empty()

    # Live debug panel
    debug_ph.markdown(f"""
```
Verdict        : {st.session_state.verdict}
Confidence     : {st.session_state.confidence*100:.1f}%  (of verdict above)
Pulse Present  : {st.session_state.pulse_present}
BPM            : {st.session_state.bpm:.1f}
Signal Quality : {st.session_state.signal_quality}  (raw={st.session_state.dbg_sig_quality_raw:.4f})
Signal Std     : {st.session_state.dbg_sig_std:.6f}
SNR Raw        : {st.session_state.dbg_snr_raw:.4f}
Dominant Freq  : {st.session_state.dbg_dom_freq:.3f} Hz
Is Calibrating : {st.session_state.is_calibrating}
Frame Count    : {st.session_state.frame_count}
```""")


# ── Main processing loop ──────────────────────────────────────────────────
if not st.session_state.running:
    render_ui_state()
else:
    try:
        # Open capture if needed
        if st.session_state.cap is None:
            if st.session_state.mode == "live":
                st.session_state.cap = cv2.VideoCapture(st.session_state.camera_index)
            else:
                tmp = os.path.join(PROJECT_ROOT, "temp_upload.mp4")
                st.session_state.cap = cv2.VideoCapture(tmp)

        cap = st.session_state.cap
        if not cap.isOpened():
            st.error("❌ Could not open video source. Check camera index or uploaded file.")
            st.session_state.running = False
            st.session_state.cap = None
            st.stop()

        while st.session_state.running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.release()
                st.session_state.cap = None
                st.session_state.running = False
                st.info("Video ended.")
                break

            # ── Step 1: Face detection ────────────────────────────────
            try:
                face_detected, face_bbox = st.session_state.face_detector.detect(frame)
            except Exception as exc:
                face_detected, face_bbox = False, (0, 0, 0, 0)

            if not face_detected:
                st.session_state.verdict = "UNCERTAIN"
                overlay = frame.copy()
                cv2.putText(overlay, "NO FACE DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2)
                frame_ph.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                channels="RGB", use_container_width=True)
                render_ui_state()
                time.sleep(0.033)
                continue

            x, y, w, h = face_bbox

            # ── Step 2: ROI extraction ────────────────────────────────
            try:
                roi_f, roi_c = extract_roi(frame, face_bbox)
                if roi_method == "Forehead Only":
                    roi_c = roi_f
                elif roi_method == "Cheeks Only":
                    roi_f = roi_c
                pkt = create_data_packet(frame, face_detected, face_bbox, roi_f, roi_c)
            except Exception:
                roi_f = roi_c = np.zeros((10, 10, 3), dtype=np.uint8)
                pkt = {"roi_forehead": roi_f, "roi_cheeks": roi_c}

            # ── Step 3: rPPG engine ───────────────────────────────────
            try:
                eng_out = process_signal(
                    roi_forehead=pkt["roi_forehead"],
                    roi_cheeks=pkt["roi_cheeks"],
                    frame_count=st.session_state.frame_count,
                    buffer_size=buf_size_val,
                )
                rppg_signal   = np.array(eng_out.get("rppg_signal", []), dtype=np.float64)
                estimated_bpm = float(eng_out.get("estimated_bpm", 0.0))
                is_cal        = bool(eng_out.get("is_calibrating", True))
                sig_quality   = float(eng_out.get("signal_quality", 0.0))
            except Exception as exc:
                rppg_signal   = np.zeros(10)
                estimated_bpm = 0.0
                is_cal        = True
                sig_quality   = 0.0

            st.session_state.is_calibrating        = is_cal
            st.session_state.dbg_sig_std            = float(np.std(rppg_signal))
            st.session_state.dbg_sig_quality_raw    = sig_quality

            # ── Step 4: Threat analysis ───────────────────────────────
            try:
                result = analyze_threat(
                    rppg_signal, estimated_bpm, is_cal, sig_quality
                )
            except Exception:
                result = {"verdict": "UNCERTAIN", "bpm": 0.0, "confidence": 0.0,
                          "pulse_present": False, "loop_detected": False,
                          "snr_score": 0.0, "dominant_freq_hz": 0.0}

            # ── Step 5: Update session state ──────────────────────────
            v = result.get("verdict", "UNCERTAIN") if not is_cal else "CALIBRATING"
            st.session_state.verdict       = v
            st.session_state.bpm           = result.get("bpm", 0.0)
            st.session_state.confidence    = result.get("confidence", 0.0)
            st.session_state.loop_detected = result.get("loop_detected", False)
            st.session_state.pulse_present = result.get("pulse_present", False)

            sq_raw = sig_quality
            sq_str = "HIGH" if sq_raw > 0.7 else "MEDIUM" if sq_raw > 0.4 else "LOW"
            st.session_state.signal_quality     = sq_str
            st.session_state.rppg_signal        = rppg_signal.tolist()
            st.session_state.dbg_snr_raw        = result.get("snr_score", 0.0)
            st.session_state.dbg_dom_freq       = result.get("dominant_freq_hz", 0.0)
            st.session_state.dominant_freq      = result.get("dominant_freq_hz", 0.0)
            st.session_state.snr_score          = result.get("snr_score", 0.0)
            st.session_state.autocorr_score     = float(result.get("snr_score", 0.0) / 10.0)

            # FFT for spectrum chart
            if len(rppg_signal) > 10:
                fft_out = np.abs(np.fft.rfft(rppg_signal))
                freqs   = np.fft.rfftfreq(len(rppg_signal), 1.0 / 30.0)
                st.session_state.fft_freqs = freqs.tolist()
                st.session_state.fft_power = fft_out.tolist()

            # ── Step 6: Draw overlays on frame ────────────────────────
            box_col = ((0,255,0) if v == "REAL"
                       else (0,0,255) if v == "THREAT"
                       else (0,215,255))
            cv2.rectangle(frame, (x,y), (x+w,y+h), box_col, 3)
            # Forehead ROI outline
            fh = int(h * 0.3); fw_lo = x + int(w*0.2); fw_hi = x + int(w*0.8)
            cv2.rectangle(frame, (fw_lo, y), (fw_hi, y+fh), (10,122,117), 1)
            cv2.putText(frame, v, (x, max(12, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_col, 2)
            H, W = frame.shape[:2]
            cv2.rectangle(frame, (8, H-44), (200, H-8), (0,0,0), -1)
            cv2.putText(frame, f"BPM: {st.session_state.bpm:.0f}",
                        (12, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            curr_fps = 1.0 / max(time.time() - t0, 1e-4)
            cv2.putText(frame, f"FPS:{curr_fps:.0f}",
                        (W-90, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)

            res_ph.metric("Resolution", f"{W}×{H}")
            fps_ph.metric("FPS", f"{curr_fps:.0f}")
            fc_ph.metric("Frames",  f"{st.session_state.frame_count}")

            # Session log (every ~30 frames after calibration)
            if (not is_cal
                    and st.session_state.frame_count > 0
                    and st.session_state.frame_count % 30 == 0):
                ts = datetime.datetime.now().strftime("%M:%S")
                entry = (f"[{ts}]  {v:<9}"
                         f" — {st.session_state.bpm:3.0f} BPM"
                         f" — {st.session_state.confidence*100:3.0f}% {v}")
                st.session_state.session_log.append(entry)

            render_ui_state()
            st.session_state.frame_count += 1
            time.sleep(0.033)

    except Exception as exc:
        st.error(f"❌ Pipeline error: {exc}")
        st.session_state.running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
