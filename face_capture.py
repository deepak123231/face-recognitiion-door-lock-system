import cv2
import os

# Prompt for user ID
user_id = input("Enter numeric user ID: ")

# Initialize camera and face detector
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0
os.makedirs("dataset", exist_ok=True)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = f"dataset/User.{user_id}.{count}.jpg"
        cv2.imwrite(file_path, face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('image', frame)

    # Break if 'q' is pressed or 50 images are captured
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count >= 50:
        break

print("\n[INFO] Done capturing faces.")
cam.release()
cv2.destroyAllWindows()
