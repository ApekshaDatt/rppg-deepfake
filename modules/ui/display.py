import cv2

def draw_overlay(frame, mode, bpm, confidence):

    # COLOR STATES
    if mode == "REAL":

        box_color = (0,255,0)

    elif mode == "THREAT":

        box_color = (0,0,255)

    else:

        box_color = (0,255,255)

    # FACE DETECTOR
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

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

        # VERDICT LABEL
        cv2.putText(
            frame,
            mode,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            box_color,
            2
        )

        # BPM
        cv2.putText(
            frame,
            f"BPM: {bpm}",
            (x, y+h+30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

        # CONFIDENCE
        cv2.putText(
            frame,
            f"CONF: {confidence}",
            (x, y+h+60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

    return frame