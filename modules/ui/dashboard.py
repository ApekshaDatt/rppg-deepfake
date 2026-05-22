import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logger import log_result

# Pipeline Modules
from config import BUFFER_SIZE, FPS, CALIBRATION_FRAMES, VERDICT_REAL, VERDICT_THREAT, VERDICT_UNCERTAIN
from modules.face_detection import FaceDetector, extract_roi, create_data_packet
from modules.rppg_engine import process_signal
from modules.threat_analyzer import analyze_threat

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="PulseShield AI", layout="wide")

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>
body { background-color: #0b1220; }
.main { background-color: #0b1220; color: white; }
h1, h2, h3, p, div { color: white !important; font-family: 'Segoe UI', sans-serif; }
[data-testid="stMetric"] {
    background-color: #111827; padding: 18px; border-radius: 14px;
    border: 1px solid #1f2937; box-shadow: 0px 0px 12px rgba(0,0,0,0.35); text-align: center;
}
[data-testid="stMetricLabel"] { color: #9ca3af !important; }
[data-testid="stMetricValue"] { color: white !important; font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER & STATUS CARDS
# -----------------------------------
st.title("🛡️ PulseShield AI")
st.subheader("Biometric Physics for Deepfake Detection")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("System Status", "ACTIVE")
col2.metric("Threat Level", "LIVE ANALYSIS")
col3.metric("Monitoring", "LIVE")
st.markdown("---")

# -----------------------------------
# MAIN LAYOUT
# -----------------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()

with right:
    st.subheader("Threat Analysis")
    m1, m2, m3 = st.columns(3)
    bpm_placeholder = m1.empty()
    conf_placeholder = m2.empty()
    pulse_placeholder = m3.empty()
    
    verdict_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("Live rPPG Signal")
    plot_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("System Logs")
    log_placeholder = st.empty()

# -----------------------------------
# PIPELINE INITIALIZATION
# -----------------------------------
face_detector = FaceDetector()
frame_count = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Camera not detected")
    st.stop()

# -----------------------------------
# CONTINUOUS LOOP
# -----------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect Face
    face_detected, face_bbox = face_detector.detect(frame)
    x, y, w, h = face_bbox

    # 2. Extract ROI
    roi_forehead, roi_cheeks = extract_roi(frame, face_bbox)

    # 3. Create DataPacket
    packet = create_data_packet(frame, face_detected, face_bbox, roi_forehead, roi_cheeks)

    # 4. Extract Signal
    rppg_output = process_signal(
        roi_forehead=packet["roi_forehead"],
        roi_cheeks=packet["roi_cheeks"],
        frame_count=frame_count,
        buffer_size=BUFFER_SIZE
    )
    rppg_signal = rppg_output["rppg_signal"]
    estimated_bpm = rppg_output["estimated_bpm"]
    is_calibrating = rppg_output["is_calibrating"]

    signal_quality = rppg_output.get("signal_quality", 1.0)

    # 5. Analyze Threat
    verdict_dict = analyze_threat(rppg_signal, estimated_bpm, is_calibrating, signal_quality)
    
    v = verdict_dict.get("verdict", VERDICT_UNCERTAIN)
    b = verdict_dict.get("bpm", 0.0)
    c = verdict_dict.get("confidence", 0.0)
    p = verdict_dict.get("pulse_present", False)

    # 6. UI Updates
    if is_calibrating:
        bpm_str = "CALIBRATING"
        conf_str = "..."
        pulse_str = "..."
        box_color = (0, 255, 255)
        graph_color = "#eab308"
        verdict_html = "<h2 style='color:#eab308;'>CALIBRATING...</h2>"
        log_text = "[INFO] Gathering baseline data...\n[INFO] Please keep face still."
    else:
        bpm_str = f"{b:.1f} BPM" if b > 0 else "N/A"
        conf_str = f"{c*100:.0f}%"
        
        if v == VERDICT_REAL:
            pulse_str = "STABLE"
            box_color = (0, 255, 0)
            graph_color = "#22c55e"
            verdict_html = "<h2 style='color:#22c55e;'>VERDICT: REAL</h2>"
            log_text = "[INFO] Biological pulse signal extracted\n[INFO] Rhythm appears natural\n[INFO] Threat confidence low"
        elif v == VERDICT_THREAT:
            pulse_str = "UNSTABLE"
            box_color = (0, 0, 255) # Red in BGR, but we convert to RGB for UI
            graph_color = "#ef4444"
            verdict_html = "<h2 style='color:#ef4444;'>VERDICT: THREAT DETECTED</h2>"
            log_text = "[WARNING] Pulse signal unstable\n[ALERT] Synthetic periodicity detected\n[CRITICAL] Deepfake threat suspected"
        else:
            pulse_str = "WEAK"
            box_color = (0, 255, 255) # Yellow
            graph_color = "#eab308"
            verdict_html = "<h2 style='color:#eab308;'>VERDICT: UNCERTAIN</h2>"
            log_text = "[INFO] Weak biological signal\n[WARNING] Inconclusive waveform pattern"

    # Draw BBox
    if face_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
        # BGR text for OpenCV (Note: box_color is used directly)
        cv2.putText(frame, v, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # Convert to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Update Metrics
    bpm_placeholder.metric("Heart Rate", bpm_str)
    conf_placeholder.metric("Confidence", conf_str)
    pulse_placeholder.metric("Pulse Signal", pulse_str)
    verdict_placeholder.markdown(verdict_html, unsafe_allow_html=True)
    log_placeholder.code(log_text)

    # 7. Plot Graph
    fig, ax = plt.subplots(figsize=(5,2))
    # Plot recent 10 seconds (up to BUFFER_SIZE)
    t = np.linspace(0, 10, len(rppg_signal))
    # Correct red color for RGB is actually used in matplotlib
    ax.plot(t, rppg_signal, color=graph_color)
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_title("Biological Pulse Signal", color='white')
    
    plot_placeholder.pyplot(fig)
    plt.close(fig) # Prevent memory leak

    # 8. Log Result
    if not is_calibrating:
        log_result(v, bpm_str, conf_str, pulse_str)

    frame_count += 1
    time.sleep(0.01)

cap.release()