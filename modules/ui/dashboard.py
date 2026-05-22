import streamlit as st
import cv2
import time

from display import draw_overlay
from plotter import generate_signal_plot
from logger import log_result

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="PulseShield AI",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

body {
    background-color: #0b1220;
}

.main {
    background-color: #0b1220;
    color: white;
}

h1, h2, h3, p, div {
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stMetric"] {
    background-color: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.35);
    text-align: center;
}

[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------

st.title("🛡️ PulseShield AI")
st.subheader("Biometric Physics for Deepfake Detection")

st.markdown("---")

# -----------------------------------
# STATUS CARDS
# -----------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("System Status", "ACTIVE")
col2.metric("Threat Level", "LOW RISK")
col3.metric("Monitoring", "LIVE")

st.markdown("---")

# -----------------------------------
# SIDEBAR MODE
# -----------------------------------

mode = st.sidebar.selectbox(
    "Detection Mode",
    ["REAL", "THREAT", "UNCERTAIN"]
)

# -----------------------------------
# THREAT STATES
# -----------------------------------

if mode == "REAL":

    bpm = "73 BPM"
    confidence = "87%"
    pulse = "STABLE"

    verdict_html = """
    <h2 style='color:#22c55e;'>
    VERDICT: REAL
    </h2>
    """

elif mode == "THREAT":

    bpm = "N/A"
    confidence = "96%"
    pulse = "UNSTABLE"

    verdict_html = """
    <h2 style='color:#ef4444;'>
    VERDICT: THREAT DETECTED
    </h2>
    """

else:

    bpm = "68 BPM"
    confidence = "61%"
    pulse = "WEAK"

    verdict_html = """
    <h2 style='color:#eab308;'>
    VERDICT: UNCERTAIN
    </h2>
    """

# -----------------------------------
# LAYOUT
# -----------------------------------

left, right = st.columns([2,1])

# CAMERA
cap = cv2.VideoCapture(0)

video_placeholder = left.empty()

while True:

    ret, frame = cap.read()

    if not ret:
        st.error("Camera not detected")
        break

    # DRAW OVERLAY
    frame = draw_overlay(
        frame,
        mode,
        bpm,
        confidence
    )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # VIDEO PANEL
    with left:

        st.subheader("Live Video Feed")

        video_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    # RIGHT PANEL
    with right:

        st.subheader("Threat Analysis")

        st.metric("Heart Rate", bpm)
        st.metric("Confidence Score", confidence)
        st.metric("Pulse Signal", pulse)

        st.markdown(
            verdict_html,
            unsafe_allow_html=True
        )

        # LOGGING
        log_result(
            mode,
            bpm,
            confidence,
            pulse
        )

        st.markdown("---")

        st.subheader("Live rPPG Signal")

        fig = generate_signal_plot(mode)

        st.pyplot(fig)

    time.sleep(0.03)

cap.release()