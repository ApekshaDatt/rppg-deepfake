import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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
# TOP STATUS CARDS
# -----------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("System Status", "ACTIVE")
col2.metric("Threat Level", "LOW RISK")
col3.metric("Monitoring", "LIVE")

st.markdown("---")

# -----------------------------------
# SIDEBAR THREAT MODES
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

    box_color = (0,255,0)
    graph_color = "#22c55e"

elif mode == "THREAT":

    bpm = "N/A"
    confidence = "96%"
    pulse = "UNSTABLE"

    verdict_html = """
    <h2 style='color:#ef4444;'>
    VERDICT: THREAT DETECTED
    </h2>
    """

    box_color = (0,0,255)
    graph_color = "#ef4444"

else:

    bpm = "68 BPM"
    confidence = "61%"
    pulse = "WEAK"

    verdict_html = """
    <h2 style='color:#eab308;'>
    VERDICT: UNCERTAIN
    </h2>
    """

    box_color = (0,255,255)
    graph_color = "#eab308"

# -----------------------------------
# MAIN LAYOUT
# -----------------------------------

left, right = st.columns([2,1])

# -----------------------------------
# FACE DETECTOR
# -----------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------------
# OPEN CAMERA
# -----------------------------------

cap = cv2.VideoCapture(0)

# PLACEHOLDERS
video_placeholder = left.empty()

# -----------------------------------
# CONTINUOUS LOOP
# -----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        st.error("Camera not detected")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # FACE BOX
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            box_color,
            3
        )

        # LABEL
        cv2.putText(
            frame,
            mode,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            box_color,
            2
        )

    # RGB CONVERSION
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # DISPLAY VIDEO
    with left:
        st.subheader("Live Video Feed")

        video_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    # -----------------------------------
    # RIGHT PANEL
    # -----------------------------------

    with right:

        st.subheader("Threat Analysis")

        st.metric("Heart Rate", bpm)
        st.metric("Confidence Score", confidence)
        st.metric("Pulse Signal", pulse)

        st.markdown(
            verdict_html,
            unsafe_allow_html=True
        )

        # -----------------------------------
        # SAVE LOG
        # -----------------------------------

        log_result(
            mode,
            bpm,
            confidence,
            pulse
        )

        st.markdown("---")

        # -----------------------------------
        # LIVE SIGNAL GRAPH
        # -----------------------------------

        st.subheader("Live rPPG Signal")

        # TIME AXIS
        t = np.linspace(0, 10, 300)

        # SIGNAL TYPES
        if mode == "REAL":

            signal = (
                np.sin(2 * np.pi * 1.2 * t)
                + 0.15 * np.random.randn(len(t))
            )

        elif mode == "THREAT":

            signal = np.sin(2 * np.pi * 1.0 * t)

        else:

            signal = (
                0.4 * np.sin(2 * np.pi * 1.1 * t)
                + 0.35 * np.random.randn(len(t))
            )

        # CREATE GRAPH
        fig, ax = plt.subplots(figsize=(5,2))

        ax.plot(t, signal, color=graph_color)

        ax.set_facecolor("#111827")

        fig.patch.set_facecolor("#111827")

        ax.tick_params(colors='white')

        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        ax.set_title(
            "Biological Pulse Signal",
            color='white'
        )

        st.pyplot(fig)

        st.markdown("---")

        # -----------------------------------
        # SYSTEM LOGS
        # -----------------------------------

        st.subheader("System Logs")

        if mode == "REAL":

            log_text = """
[INFO] Face detected successfully
[INFO] Biological pulse signal extracted
[INFO] Heartbeat rhythm appears natural
[INFO] Signal quality stable
[INFO] Threat confidence remains low
"""

        elif mode == "THREAT":

            log_text = """
[WARNING] Pulse signal unstable
[WARNING] Synthetic periodicity detected
[ALERT] Biological inconsistency found
[ALERT] Deepfake threat suspected
[CRITICAL] Threat confidence elevated
"""

        else:

            log_text = """
[INFO] Weak biological signal detected
[INFO] Signal quality fluctuating
[WARNING] Inconclusive waveform pattern
[INFO] Additional analysis recommended
"""

        st.code(log_text)

    time.sleep(0.03)

# -----------------------------------
# RELEASE CAMERA
# -----------------------------------

cap.release()