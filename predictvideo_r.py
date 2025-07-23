from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os
from random import randrange

# Load the trained model and label binarizer
print("[INFO] Loading model and label binarizer...")
model_path = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/model/activity.keras'
label_bin_path = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/model/lb.pickle'
output_images_dir = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/images output'

model = load_model(model_path)
lb = pickle.loads(open(label_bin_path, "rb").read())

# Define thresholds
EXPLOSION_THRESHOLD = 0.9
ROAD_ACCIDENT_THRESHOLD = 0.9

# Initialize queue for prediction averaging
size = 128
Q = deque(maxlen=size)

# Mean subtraction values
mean = np.array([123.68, 116.779, 103.939], dtype="float32")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Get frame dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = original_width / original_height

display_width = 800  # Set display width

display_height = int(display_width / aspect_ratio)
print("[INFO] Display window dimensions: {}x{}".format(display_width, display_height))

cv2.namedWindow("Live Prediction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Prediction", display_width, display_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean

    # Make predictions
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    confidence = results[i]

    # Apply thresholds
    if label == "Explosion" and confidence < EXPLOSION_THRESHOLD:
        label = "Normal"
    elif label == "RoadAccident" and confidence < ROAD_ACCIDENT_THRESHOLD:
        label = "Normal"
    if label not in ["Explosion", "RoadAccident"]:
        label = "Normal"

    text_color = (0, 0, 255) if label in ["Explosion", "RoadAccident"] else (0, 255, 0)
    text = "{}: {:.2f}%".format(label, confidence * 100)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 5)

    if label in ["Explosion", "RoadAccident"]:
        alert = "Warning: {}".format(label)
        cv2.putText(output, alert, (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 5)
        filename = "{}.png".format(randrange(0, 1000))
        cv2.imwrite(os.path.join(output_images_dir, filename), output)

    output_resized = cv2.resize(output, (display_width, display_height))
    cv2.imshow("Live Prediction", output_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
