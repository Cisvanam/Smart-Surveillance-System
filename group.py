import cv2
import dlib
import numpy as np
import os
import face_recognition
from ultralytics import YOLO
from imutils import face_utils
import pandas as pd
from datetime import datetime

model = YOLO("yolov8n.pt")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

professor_encodings = []
for file in os.listdir("professor_faces"):
    if file.startswith('.'):
        continue
    path = os.path.join("professor_faces", file)
    img = face_recognition.load_image_file(path)
    enc = face_recognition.face_encodings(img)
    if enc:
        professor_encodings.append(enc[0])

student_encodings = []
student_names = []
for name in os.listdir("students_faces"):
    folder = os.path.join("students_faces", name)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.startswith('.'):
                continue
            path = os.path.join(folder, file)
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if enc:
                student_encodings.append(enc[0])
                student_names.append(name)


log_file = "Class Tracked Data Section 1.xlsx"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Timestamp", "Detection"]).to_excel(log_file, index=False)

def log_detection(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[timestamp, event]], columns=["Timestamp", "Detection"])
    with pd.ExcelWriter(log_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        sheet = writer.sheets['Sheet1']
        startrow = sheet.max_row
        new_row.to_excel(writer, index=False, header=False, startrow=startrow)

def detect_eye_direction(eye_img):
    try:
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "CENTER"
        c = max(contours, key=cv2.contourArea)
        x, _, w, _ = cv2.boundingRect(c)
        center_x = x + w // 2
        width = eye_img.shape[1]
        if center_x < width / 3:
            return "RIGHT"
        elif center_x > 2 * width / 3:
            return "LEFT"
        return "CENTER"
    except:
        return "CENTER"

def get_chin_direction(shape):
    left, right, chin = shape[1], shape[15], shape[8]
    center_x = (left[0] + right[0]) / 2
    deviation = chin[0] - center_x
    if deviation > 20:
        return "RIGHT"
    elif deviation < -20:
        return "LEFT"
    return "CENTER"


cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("Class Cheating Detection Section 1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

print("Monitoring started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    flagged = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    professor_present = False
    student_identified = []
    faces_info = []

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        is_professor = any(face_recognition.compare_faces([e], enc, tolerance=0.5)[0] for e in professor_encodings)
        name = "Professor" if is_professor else "Unknown"
        if is_professor:
            professor_present = True
        else:
            matches = face_recognition.compare_faces(student_encodings, enc, tolerance=0.5)
            if True in matches:
                best_match = matches.index(True)
                name = student_names[best_match]
                student_identified.append(name)

        faces_info.append((left, top, right, bottom, name))

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    faces = detector(gray)
    for i, face in enumerate(faces):
        if i >= len(faces_info):
            continue
        _, _, _, _, name = faces_info[i]
        is_professor = (name == "Professor")

        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)
        chin_dir = get_chin_direction(shape_np)

        (lx, ly, lw, lh) = cv2.boundingRect(shape_np[36:42])
        (rx, ry, rw, rh) = cv2.boundingRect(shape_np[42:48])
        left_eye_img = frame[ly:ly+lh, lx:lx+lw]
        right_eye_img = frame[ry:ry+rh, rx:rx+rw]

        gaze_left = detect_eye_direction(left_eye_img)
        gaze_right = detect_eye_direction(right_eye_img)

        if (gaze_left != "CENTER" or gaze_right != "CENTER") and chin_dir == "CENTER" and not is_professor:
            flagged = True
            log_detection(f"Gaze shift: {gaze_left}/{gaze_right}")
            cv2.putText(frame, f"Gaze: {gaze_left}/{gaze_right}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if chin_dir != "CENTER" and not is_professor:
            flagged = True
            log_detection(f"Chin turned: {chin_dir}")
            cv2.putText(frame, f"Chin: {chin_dir}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    results = model.predict(frame, conf=0.35, verbose=False)
    standing_detected = False
    laptop_present = False
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1
            label = model.names[cls]

            if cls == 0:  # person
                person_count += 1
                if height > 0.6 * frame_height and not professor_present:
                    standing_detected = True

            if cls == 63:  # laptop
                laptop_present = True

            if cls == 67:  # phone
                if not professor_present:
                    flagged = True
                    log_detection("Phone detected")
                    cv2.putText(frame, "Phone Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if person_count > 1 and not professor_present:
        flagged = True
        log_detection("Multiple people detected (excluding professor)")
        cv2.putText(frame, "Multiple People Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if standing_detected and not laptop_present and not professor_present:
        flagged = True
        log_detection("Student standing without laptop")
        cv2.putText(frame, "Cheating: Standing w/o Laptop", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not flagged:
        cv2.putText(frame, "No Cheating Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Cheating Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()