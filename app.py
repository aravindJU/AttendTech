import os
import cv2
import sqlite3
import shutil
import time
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_NAME = "attendance.db"
FACE_DIR = "faces"
MODEL_FILE = "trainer.yml"

os.makedirs(FACE_DIR, exist_ok=True)

# ----------------- DATABASE -----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS Students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll_no TEXT UNIQUE NOT NULL)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS Attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY(student_id) REFERENCES Students(id))''')
    conn.commit()
    conn.close()

def add_student(name, roll_no):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO Students (name, roll_no) VALUES (?, ?)", (name, roll_no))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_students():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Students")
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_student(student_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT roll_no, name FROM Students WHERE id=?", (student_id,))
    student = cur.fetchone()
    if student:
        roll_no, name = student
        cur.execute("DELETE FROM Students WHERE id=?", (student_id,))
        conn.commit()
        conn.close()
        return roll_no, name
    conn.close()
    return None, None

def mark_attendance(student_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Attendance WHERE student_id=?", (student_id,))
    record = cur.fetchone()
    if record:
        conn.close()
        return  # prevent duplicates
    cur.execute("INSERT INTO Attendance (student_id, timestamp) VALUES (?, ?)", 
                (student_id, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def get_attendance_logs():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''SELECT Attendance.id, Students.name, Students.roll_no, Attendance.timestamp
                   FROM Attendance
                   INNER JOIN Students ON Attendance.student_id = Students.id
                   ORDER BY Attendance.id DESC''')
    logs = cur.fetchall()
    conn.close()
    return logs

def clear_attendance():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM Attendance")
    conn.commit()
    conn.close()

# ----------------- FACE FOLDER SAFE DELETE -----------------
def safe_delete_folder(folder_path):
    cv2.destroyAllWindows()  # Close all OpenCV windows
    for i in range(3):  # Retry a few times
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            return True
        except PermissionError:
            time.sleep(0.5)
    return False

# ----------------- ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        roll_no = request.form["roll_no"].strip()
        if not name or not roll_no:
            flash("Name and Roll Number required", "danger")
            return redirect(url_for("register"))

        student_folder = os.path.join(FACE_DIR, roll_no)
        os.makedirs(student_folder, exist_ok=True)

        # Capture face
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{student_folder}/{count}.jpg", face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.imshow("Register Face - Press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
                break
        cap.release()
        cv2.destroyAllWindows()

        if add_student(name, roll_no):
            flash(f"Student {name} registered!", "success")
        else:
            flash("Roll number already exists!", "danger")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/students")
def students_page():
    students = get_students()
    return render_template("students.html", students=students)

@app.route("/delete/<int:student_id>", methods=["POST"])
def delete_student_page(student_id):
    roll_no, name = delete_student(student_id)
    if roll_no:
        folder = os.path.join(FACE_DIR, roll_no)
        if os.path.exists(folder):
            if safe_delete_folder(folder):
                flash(f"Deleted student: {name}", "success")
            else:
                flash(f"Could not delete face folder for {name}. Close any webcam windows first.", "danger")
        else:
            flash(f"Deleted student: {name}", "success")
    else:
        flash("Student not found", "danger")
    return redirect(url_for("students_page"))

@app.route("/train")
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []

    for student in get_students():
        student_id = student[0]
        roll_no = student[2]
        folder = os.path.join(FACE_DIR, roll_no)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(student_id)

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_FILE)
        flash("Model trained successfully!", "success")
    else:
        flash("No faces to train. Register students first.", "warning")
    return redirect(url_for("index"))

@app.route("/mark_attendance")
def mark_attendance_page():
    if not os.path.exists(MODEL_FILE):
        flash("Train model first!", "warning")
        return redirect(url_for("index"))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    cap = cv2.VideoCapture(0)
    recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                mark_attendance(id_)
                recognized = True
                cv2.putText(frame, "Attendance Marked!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                break

        cv2.imshow("Mark Attendance", frame)
        if recognized or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if recognized:
        flash("Attendance marked!", "success")
    else:
        flash("No face recognized.", "warning")
    return redirect(url_for("attendance_log"))

@app.route("/attendance")
def attendance_log():
    logs = get_attendance_logs()
    return render_template("attendance.html", logs=logs)

@app.route("/clear_attendance", methods=["POST"])
def clear_attendance_route():
    clear_attendance()
    flash("Attendance logs cleared!", "success")
    return redirect(url_for("attendance_log"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
