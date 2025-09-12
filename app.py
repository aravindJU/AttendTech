def recognize_faces():
    if not os.path.exists(MODEL_FILE):
        return "No trained model found. Please train first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    message = "No known face detected."

    # Keep capturing frames until we detect at least one face
    recognized = False
    while not recognized:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first face only
            face_img = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face_img)

            if conf < 80:
                log_attendance(id_)
                recognized = True

                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT name FROM students WHERE id=?", (id_,))
                student = cur.fetchone()
                conn.close()

                if student:
                    name = student["name"]
                    message = f"Attendance logged for {name}."
                    cv2.putText(frame, f"{name}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the recognized frame briefly
                cv2.imshow("Recognizing Face", frame)
                cv2.waitKey(500)
            break  # Exit the while loop immediately

    cam.release()
    cv2.destroyAllWindows()
    return message
