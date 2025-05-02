import cv2
import serial
import time

# Load trained model and Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Connect to Arduino
try:
    arduino = serial.Serial('COM3', 9600)  # Replace COM3 with your actual port
    time.sleep(2)  # Wait for connection to initialize
    print("[INFO] Connected to Arduino.")
except:
    arduino = None
    print("[WARNING] Arduino not connected.")

# Define font for text
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 50:
            label = f"User {id}"
            color = (0, 255, 0)
            if arduino:
                arduino.write(b'U')  # Send unlock command
                print("[INFO] Unlock signal sent to Arduino.")
                time.sleep(5)        # Wait before allowing another unlock
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x+5, y-5), font, 1, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
