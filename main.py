import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# Folder containing known employee images
images_path = "images"
known_encodings, known_names = [], []

print("🔍 Loading employee images...")

for filename in os.listdir(images_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        path = os.path.join(images_path, filename)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"⚠️ Cannot read {filename}, skipped.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"✅ Loaded and encoded: {name}")
        else:
            print(f"⚠️ No face detected in {filename}, skipped.")

print(f"\nTotal known employees: {len(known_names)}")

# ===== Attendance Logging Setup =====
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Match %"])

# Helper function to check if today's attendance is already marked
def already_marked_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(attendance_file):
        return False
    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == today:
                return True
    return False

# ===== Start Camera =====
camera = cv2.VideoCapture(0)
print("\n🎥 Camera started. Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        match_percentage = (1 - face_distances[best_match_index]) * 100

        if matches[best_match_index]:
            name = known_names[best_match_index]
            color = (0, 255, 0)

            # ===== Attendance Logging =====
            if not already_marked_today(name):
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")
                with open(attendance_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, date, time, f"{match_percentage:.2f}%"])
                print(f"🟢 Attendance marked for {name} at {time}")
            else:
                print(f"ℹ️ {name}'s attendance already marked today.")

        else:
            color = (0, 0, 255)

        # Scale coordinates back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, f"{name} ({match_percentage:.1f}%)", (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Employee Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("👋 Camera closed. Attendance saved in attendance.csv")
