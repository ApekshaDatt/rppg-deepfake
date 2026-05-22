import csv
import os
from datetime import datetime

LOG_FILE = "logs.csv"

def log_result(verdict, bpm, confidence, pulse):

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode='a', newline='') as file:

        writer = csv.writer(file)

        # HEADER
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Verdict",
                "BPM",
                "Confidence",
                "Pulse"
            ])

        # DATA ROW
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            verdict,
            bpm,
            confidence,
            pulse
        ])