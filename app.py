import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import plotly.graph_objects as go
from collections import deque
import random
from modules.rppg import RPPGExtractor
from modules.ai_engine import AIEngineMock

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PulseShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Poppins:wght@500;600&display=swap');

    /* Global settings */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0B0F19;
        color: #E5E7EB;
    }

    /* Container adjustments for wide layout */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 95% !important;
    }

    /* Cards */
    div.stApp > header {
        background-color: transparent !important;
    }
    
    .panel {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .header-panel {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        color: #E5E7EB !important;
    }
    
    .title {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        color: #06B6D4 !important;
    }
    .subtitle {
        font-size: 0.9rem;
        color: #9CA3AF;
        font-weight: 300;
    }

    /* Metric Labels */
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
    .metric-value.cyan { color: #06B6D4; }
    .metric-value.red { color: #EF4444; }
    .metric-value.green { color: #10B981; }

    /* Verdict Tag */
    .verdict-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1.2rem;
        letter-spacing: 0.1em;
    }
    .verdict-live {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .verdict-deepfake {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid #EF4444;
        animation: pulse-red 2s infinite;
    }
    .verdict-uncertain {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Footer */
    .fixed-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0B0F19;
        border-top: 1px solid #1f2937;
        padding: 0.5rem 2rem;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        font-size: 0.75rem;
        color: #6B7280;
        z-index: 1000;
    }
    .footer-item {
        margin-right: 2rem;
        display: flex;
        align-items: center;
    }
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #06B6D4;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- WEBRTC PROCESSOR ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Modules
        self.rppg = RPPGExtractor(fps=30.0, buffer_size=150)
        self.ai = AIEngineMock()
        
        # Public states for Streamlit UI
        self.bpm = 0.0
        self.risk = 50.0
        self.integrity = 50.0
        self.signal_data = []
        self.latency = 0
        self.logs = []
        
        self._add_log("Stream initialized. Awaiting video feed.")
        
    def _add_log(self, msg):
        t = time.strftime("[%H:%M:%S]")
        self.logs.append(f"{t} {msg}")
        if len(self.logs) > 10:
            self.logs.pop(0)

    def transform(self, frame):
        start_proc = time.time()
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # FPS Calculation
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.rppg.fps = self.fps if self.fps > 0 else 30.0 # update dynamic FPS
            self.frame_count = 0
            self.start_time = time.time()
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        # Dark dashboard overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (11, 15, 25), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        
        # Extract rPPG if face found
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                roi_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
                # Draw landmarks
                roi_pixels = []
                for idx in roi_points:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    # Extract green channel value from original un-darkened image
                    if 0 <= y < h and 0 <= x < w:
                        g_val = img_rgb[y, x, 1] 
                        roi_pixels.append(g_val)
                    cv2.circle(img, (x, y), 1, (212, 182, 6), -1) 
                    
                # Calculate mean green value for rPPG
                if roi_pixels:
                    mean_g = np.mean(roi_pixels)
                    self.rppg.add_frame_data(mean_g, time.time())
                    
                # Bounding box
                x_min = min([int(lm.x * w) for lm in face_landmarks.landmark])
                x_max = max([int(lm.x * w) for lm in face_landmarks.landmark])
                y_min = min([int(lm.y * h) for lm in face_landmarks.landmark])
                y_max = max([int(lm.y * h) for lm in face_landmarks.landmark])
                
                length = 20
                thickness = 2
                color = (212, 182, 6)
                for pt1, pt2 in [
                    ((x_min, y_min), (x_min + length, y_min)), ((x_min, y_min), (x_min, y_min + length)),
                    ((x_max, y_min), (x_max - length, y_min)), ((x_max, y_min), (x_max, y_min + length)),
                    ((x_min, y_max), (x_min + length, y_max)), ((x_min, y_max), (x_min, y_max - length)),
                    ((x_max, y_max), (x_max - length, y_max)), ((x_max, y_max), (x_max, y_max - length))
                ]:
                    cv2.line(img, pt1, pt2, color, thickness)
        else:
            # If no face, zero out signals to trigger deepfake/uncertain alert
            self.rppg.add_frame_data(0, time.time())

        # Process Risk and update public states
        risk, integrity = self.ai.analyze_signal(self.rppg.filtered_signal)
        self.risk = risk
        self.integrity = integrity
        self.bpm = self.rppg.current_bpm
        self.signal_data = self.rppg.filtered_signal
        
        # Logging logic simulation
        if len(self.logs) == 1 and results.multi_face_landmarks:
            self._add_log("Target localized. FaceMesh active.")
        if len(self.logs) == 2 and len(self.signal_data) > 30:
            self._add_log("Extracting baseline biological signal...")
        if len(self.logs) == 3 and len(self.signal_data) >= 150:
            if integrity > 70:
                self._add_log("Periodicity within expected human thresholds.")
            else:
                self._add_log("WARNING: Periodicity anomalous or missing.")

        # Scanning line overlay
        scan_y = int((time.time() % 3) / 3 * h)
        cv2.line(img, (0, scan_y), (w, scan_y), (212, 182, 6), 1)

        self.latency = int((time.time() - start_proc) * 1000)
        return img


# --- HTML RENDER TEMPLATES ---
def render_verdict(risk):
    if risk < 35:
        verdict, v_class = "LIVE", "verdict-live"
    elif risk > 65:
        verdict, v_class = "DEEPFAKE", "verdict-deepfake"
    else:
        verdict, v_class = "UNCERTAIN", "verdict-uncertain"
    return f'<div style="margin-top: 1rem; text-align: center;"><span class="verdict-tag {v_class}">STATUS: {verdict}</span></div>'

def render_metric(label, value, is_percent=False, invert_color=False):
    if is_percent:
        color = "cyan" if (value > 50 and not invert_color) or (value <= 50 and invert_color) else "red"
        val_str = f"{int(value)}%"
    else:
        color = "cyan"
        val_str = f"{int(value)}"
    return f"""<div class="metric-label">{label}</div><div class="metric-value {color}">{val_str}</div>"""

def render_risk_meter(risk):
    color = "#EF4444" if risk > 50 else "#06B6D4"
    text_color = "red" if risk > 50 else "cyan"
    return f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.5rem;">
        <div class="metric-label">Authenticity Risk</div>
        <div class="metric-value {text_color}">{int(risk)}%</div>
    </div>
    <div style="width: 100%; background-color: #1f2937; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 2rem;">
        <div style="width: {risk}%; background-color: {color}; height: 100%;"></div>
    </div>
    """

def render_logs(logs):
    log_html = "<div style='height: 150px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; color: #9CA3AF; background-color: #0B0F19; padding: 1rem; border-radius: 4px; border: 1px solid #1f2937;'>"
    for log in reversed(logs):
        log_html += f"<div style='margin-bottom: 0.25rem;'>{log}</div>"
    log_html += "</div>"
    return log_html

def render_footer(fps, latency):
    return f"""
    <div class="fixed-footer">
        <div class="footer-item">FPS: {int(fps)}</div>
        <div class="footer-item">Latency: {latency}ms</div>
        <div class="footer-item"><span class="status-dot"></span> Engine: ACTIVE</div>
    </div>
    """


# --- UI LAYOUT ---

st.markdown("""
<div class="header-panel">
    <div>
        <div class="title">PulseShield AI</div>
        <div class="subtitle">Biometric Physics for Deepfake Detection</div>
    </div>
    <div style="color: #06B6D4; font-size: 0.9rem; display: flex; align-items: center;">
        <span class="status-dot"></span> SYSTEM STATUS: ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### LIVE SENSOR FEED")
    
    webrtc_ctx = webrtc_streamer(
        key="pulseshield",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    verdict_ph = st.empty()
    st.markdown("<hr style='border-color: #1f2937; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        bpm_ph = st.empty()
    with m2:
        integrity_ph = st.empty()
        
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    risk_ph = st.empty()
    
    st.markdown("### LIVE PULSE WAVEFORM")
    graph_ph = st.empty()
    
    st.markdown("<hr style='border-color: #1f2937; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("### SYSTEM LOGS")
    logs_ph = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

footer_ph = st.empty()

# Initialize Placeholders
verdict_ph.markdown(render_verdict(50), unsafe_allow_html=True)
bpm_ph.markdown(render_metric("Estimated BPM", 0), unsafe_allow_html=True)
integrity_ph.markdown(render_metric("Biological Signal Integrity", 0, True), unsafe_allow_html=True)
risk_ph.markdown(render_risk_meter(50), unsafe_allow_html=True)
logs_ph.markdown(render_logs(["[System] Ready. Awaiting feed..."]), unsafe_allow_html=True)
footer_ph.markdown(render_footer(0, 0), unsafe_allow_html=True)

fig = go.Figure()
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=10, b=0), height=200,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='#1f2937', showticklabels=False, zeroline=False, range=[-2, 2])
)
graph_ph.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- ASYNC UPDATE LOOP ---
if webrtc_ctx.state.playing:
    while True:
        if webrtc_ctx.video_processor:
            vp = webrtc_ctx.video_processor
            
            # Update UI from VideoProcessor attributes
            verdict_ph.markdown(render_verdict(vp.risk), unsafe_allow_html=True)
            bpm_ph.markdown(render_metric("Estimated BPM", vp.bpm), unsafe_allow_html=True)
            integrity_ph.markdown(render_metric("Biological Signal Integrity", vp.integrity, True), unsafe_allow_html=True)
            risk_ph.markdown(render_risk_meter(vp.risk), unsafe_allow_html=True)
            logs_ph.markdown(render_logs(vp.logs), unsafe_allow_html=True)
            footer_ph.markdown(render_footer(vp.fps, vp.latency), unsafe_allow_html=True)
            
            # Update Plotly Waveform
            sig = vp.signal_data
            if sig and len(sig) > 0:
                y_vals = sig
                x_vals = list(range(len(y_vals)))
                
                # Dynamic y-axis based on signal min/max
                y_min, y_max = min(y_vals), max(y_vals)
                margin = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines', line=dict(color='#06B6D4', width=2),
                    fill='tozeroy', fillcolor='rgba(6, 182, 212, 0.1)'
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0), height=200,
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='#1f2937', showticklabels=False, zeroline=False, range=[y_min - margin, y_max + margin])
                )
                graph_ph.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
        time.sleep(0.5)
