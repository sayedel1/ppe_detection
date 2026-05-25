"""
PPE Detection System using YOLOv8 — Custom Trained Model + Dashboard
======================================================================
بيكتشف: Cap, Gloves, Googles, No-Cap, Person, White_C
وبيعرض Dashboard جنب الفيديو بعدد الـ Safe والـ Violations

Installation:
    pip install ultralytics opencv-python

Usage:
    python ppe_detection.py                  # webcam
    python ppe_detection.py --source ccp.mp4
    python ppe_detection.py --source image.jpg

تأكد إن best.pt في نفس الفولدر بتاع السكريبت
"""

import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# =====================
# Settings
# =====================
CONFIDENCE_THRESHOLD = 0.5
ALERT_COLOR  = (0, 0, 255)    # RED
SAFE_COLOR   = (0, 200, 0)    # GREEN
INFO_COLOR   = (255, 200, 0)  # YELLOW
DASH_BG      = (20, 20, 20)   # Dashboard background

PPE_CLASSES = {
    "cap":     {"label": "Cap"},
    "gloves":  {"label": "Gloves"},
    "googles": {"label": "Googles"},
    "no-cap":  {"label": "NO Cap"},
    "person":  {"label": "Person"},
    "white_c": {"label": "White Coat"},
}

VIOLATION_CLASSES = {"no-cap"}
SAFE_PPE_CLASSES  = {"cap", "gloves", "googles", "white_c"}


# =====================
# Draw Bounding Box
# =====================
def draw_box(frame, box, label, color, conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)


# =====================
# Draw Dashboard Panel
# =====================
def draw_dashboard(h, safe_count, violation_count, total_persons, frame_count):
    dash_w = 400
    dash = np.zeros((h, dash_w, 3), dtype=np.uint8)
    dash[:] = DASH_BG

    # Title
    cv2.putText(dash, "PPE MONITOR", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.line(dash, (20, 65), (dash_w - 20, 65), (80, 80, 80), 1)

    # Time
    time_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(dash, time_str, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)

    # ---- Stats ----
    y = 160

    # Total Persons
    cv2.putText(dash, "Total Persons", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(dash, str(total_persons), (20, y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    y += 110
    cv2.line(dash, (20, y), (dash_w - 20, y), (60, 60, 60), 1)
    y += 30

    # Safe
    cv2.rectangle(dash, (20, y), (dash_w - 20, y + 80), (0, 60, 0), -1)
    cv2.putText(dash, "SAFE (With PPE)", (30, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 1)
    cv2.putText(dash, str(safe_count), (30, y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, SAFE_COLOR, 3)
    y += 100

    # Violations
    cv2.rectangle(dash, (20, y), (dash_w - 20, y + 80), (60, 0, 0), -1)
    cv2.putText(dash, "VIOLATION (No PPE)", (30, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 200), 1)
    cv2.putText(dash, str(violation_count), (30, y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, ALERT_COLOR, 3)
    y += 110

    cv2.line(dash, (20, y), (dash_w - 20, y), (60, 60, 60), 1)
    y += 30

    # Frame counter
    cv2.putText(dash, f"Frame: {frame_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)

    # Overall status
    y += 50
    if violation_count == 0 and total_persons > 0:
        status_text = "ALL SAFE"
        status_color = SAFE_COLOR
        bg_color = (0, 50, 0)
    elif violation_count > 0:
        status_text = "ALERT!"
        status_color = ALERT_COLOR
        bg_color = (50, 0, 0)
    else:
        status_text = "NO DETECT"
        status_color = INFO_COLOR
        bg_color = (40, 40, 0)

    cv2.rectangle(dash, (20, y), (dash_w - 20, y + 60), bg_color, -1)
    cv2.rectangle(dash, (20, y), (dash_w - 20, y + 60), status_color, 2)
    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3), 0
    cv2.putText(dash, status_text, (30, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    return dash


# =====================
# Process Frame
# =====================
def process_frame(frame, model):
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    violations    = 0
    safe_ppe      = 0
    total_persons = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names[cls_id].lower()

        if label == "person":
            total_persons += 1
            color = INFO_COLOR
        elif label in VIOLATION_CLASSES:
            color = ALERT_COLOR
            violations += 1
        elif label in SAFE_PPE_CLASSES:
            color = SAFE_COLOR
            safe_ppe += 1
        else:
            color = INFO_COLOR

        display_label = PPE_CLASSES.get(label, {}).get("label", label)
        draw_box(frame, box.xyxy[0], display_label, color, conf)

    return frame, violations, safe_ppe, total_persons


# =====================
# Main Loop
# =====================
def run(source):
    print("Loading YOLOv8 PPE model...")
    model = YOLO("best.pt")
    print("Model loaded. Press Q to quit.\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {source} | {w}x{h} | FPS: {fps:.0f}")

    frame_count = 0

    cv2.namedWindow("PPE Detection System", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        frame_count += 1
        annotated, violations, safe_ppe, total_persons = process_frame(frame, model)

        # resize الفيديو عشان يتناسب مع الشاشة
        display_h = 720
        scale = display_h / h
        display_w = int(w * scale)
        resized = cv2.resize(annotated, (display_w, display_h))

        # Dashboard
        dashboard = draw_dashboard(display_h, safe_ppe, violations, total_persons, frame_count)

        # دمج الفيديو مع الـ Dashboard
        combined = np.hstack([resized, dashboard])

        cv2.imshow("PPE Detection System", combined)

        if violations > 0:
            print(f"Frame {frame_count}: {violations} violation(s)!")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {frame_count} frames processed.")


# =====================
# Entry Point
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Detection with Dashboard")
    parser.add_argument("--source", default=0,
                        help="0 for webcam, or path to video/image")
    args = parser.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run(source)