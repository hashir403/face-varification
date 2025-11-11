# 👁️‍🗨️ Real-Time Face Verification & Attendance System (Python + OpenCV + Face Recognition)

This project is a **real-time face verification and attendance system** built using **Python, OpenCV, and the Face Recognition library (dlib)**.  
It automatically detects and verifies employee faces through a webcam feed and records attendance in a CSV file — ensuring each employee’s attendance is marked **only once per day**.

---

## 🚀 Features

✅ Real-time face detection and recognition using webcam  
✅ Automatic employee verification from pre-saved images  
✅ Green box for verified employees, red for unknown persons  
✅ Displays **match percentage** on screen  
✅ Automatically logs attendance (Name, Date, Time, Match %) in `attendance.csv`  
✅ Prevents duplicate attendance entries within the same day  
✅ Lightweight, fast, and works fully offline  

---

## 🧠 How It Works

1. Add your employee photos in the `images/` folder.  
   - File names represent employee names (e.g. `hashir.jpg`, `asif.png`).  
2. Run the script:
   ```bash
   python main.py

--------
Requirements

Python 3.10
OpenCV
dlib
face_recognition
NumPy
Pillow
Install all dependencies:   pip install -r requirements.txt
