import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from imutils import face_utils
import pandas as pd
from datetime import datetime
import os
import socket


log_file = "Tracked Data.xlsx"
video_file = "cheating_detection_output.mp4"


model = YOLO("yolov8n.pt")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


if not os.path.exists(log_file):
    df_init = pd.DataFrame(columns=["Timestamp", "Detection"])
    df_init.to_excel(log_file, index=False)


def log_detection(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[timestamp, event]], columns=["Timestamp", "Detection"])
    try:
        with pd.ExcelWriter(log_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            sheet = writer.sheets['Sheet1']
            startrow = sheet.max_row
            new_row.to_excel(writer, index=False, header=False, startrow=startrow)
    except Exception as e:
        print(f"Logging failed: {e}")


def detect_eye_direction(eye_img):
    try:
        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.equalizeHist(gray_eye)
        blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return "CENTER"
        contour = max(contours, key=cv2.contourArea)
        (x, _, w, _) = cv2.boundingRect(contour)
        center_x = x + w // 2
        width = eye_img.shape[1]
        if center_x < width / 3:
            return "RIGHT"
        elif center_x > 2 * width / 3:
            return "LEFT"
        else:
            return "CENTER"
    except:
        return "CENTER"


def get_chin_direction(shape):
    left = shape[1]
    right = shape[15]
    chin = shape[8]
    center_x = (left[0] + right[0]) / 2
    deviation = chin[0] - center_x
    if deviation > 20:
        return "RIGHT"
    elif deviation < -20:
        return "LEFT"
    return "CENTER"


hostname = socket.gethostname().replace(" ", "").replace("'", "")
today = datetime.now().strftime("%Y-%m-%d")
video_filename = f"{today}_{hostname}.mp4"
log_file = f"{today}_{hostname}.xlsx"


if not os.path.exists(log_file):
    pd.DataFrame(columns=["Timestamp", "Detection"]).to_excel(log_file, index=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not access webcam.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))


print("Personal cheating detection running... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flagged = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dlib facial detection
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        chin_dir = get_chin_direction(shape_np)

        (lx, ly, lw, lh) = cv2.boundingRect(shape_np[36:42])
        (rx, ry, rw, rh) = cv2.boundingRect(shape_np[42:48])

        left_eye_img = frame[ly:ly+lh, lx:lx+lw]
        right_eye_img = frame[ry:ry+rh, rx:rx+rw]

        gaze_left = detect_eye_direction(left_eye_img)
        gaze_right = detect_eye_direction(right_eye_img)

        if (gaze_left != "CENTER" or gaze_right != "CENTER") and chin_dir == "CENTER":
            flagged = True
            cv2.putText(frame, f"Gaze: {gaze_left}/{gaze_right}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            log_detection(f"Gaze shift: {gaze_left}/{gaze_right}")

        if chin_dir != "CENTER":
            flagged = True
            cv2.putText(frame, f"Chin Direction: {chin_dir}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            log_detection(f"Chin turned: {chin_dir}")


    results = model.predict(frame, conf=0.35, classes=[0, 67], verbose=False)
    for r in results:
        person_count = 0
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if cls == 67 else (255, 0, 0)

            if label == 'person':
                person_count += 1
            elif label == 'cell phone':
                flagged = True
                log_detection("Phone detected in frame")
                cv2.putText(frame, "Phone Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if person_count > 1:
            flagged = True
            log_detection("Multiple people detected")
            cv2.putText(frame, "Multiple People Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not flagged:
        cv2.putText(frame, "No Cheating Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Personal Cheating Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()