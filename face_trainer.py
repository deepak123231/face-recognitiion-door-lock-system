import cv2
import os
import numpy as np
from PIL import Image

# Path to dataset folder
dataset_path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to get images and labels from dataset
def get_images_and_labels(path):
    face_samples = []
    ids = []

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')  # grayscale
        img_numpy = np.array(gray_img, 'uint8')

        # Extract ID from filename (format: User.ID.num.jpg)
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids

print("[INFO] Training faces. Please wait...")

faces, ids = get_images_and_labels(dataset_path)
recognizer.train(faces, np.array(ids))
recognizer.save('trained_model.yml')

print("[INFO] Training complete. Model saved as 'trained_model.yml'")
