# Import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os
from random import randrange
import winsound
import pyttsx3
import smtplib
from email.message import EmailMessage

# Paths
img_dir = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/images output'
model_path = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/model/activity.keras'
label_bin = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/model/lb.pickle'
input_dir = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/testing videos/e1.mp4'
output_dir = 'C:/Users/mikki/Downloads/Explosion-and-RoadAccident-detection-using-Deep-Learning/output/outputvideo.avi'
size = 128

# Define suspicious thresholds
EXPLOSION_THRESHOLD = 0.6
ROAD_ACCIDENT_THRESHOLD = 0.6

# Load model and label binarizer
print("[INFO] loading model and label binarizer...")
model = load_model(model_path)
lb = pickle.loads(open(label_bin, "rb").read())

# Initialize mean subtraction and prediction queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=size)

# Initialize video stream
vs = cv2.VideoCapture(input_dir)
writer = None
(W, H) = (None, None)

# Get original video dimensions
original_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = original_width / original_height
display_width = 800
display_height = int(display_width / aspect_ratio)

print("[INFO] Original video dimensions: {}x{}".format(original_width, original_height))
print("[INFO] Display window dimensions: {}x{}".format(display_width, display_height))

# Create a display window
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", display_width, display_height)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Email Sending Function
def send_email(alert_type, confidence, image_path):
    sender_email = "xyzabc344@gmail.com"
    sender_password = "xghducndjn12"  # Use App Password if needed
    recipient_email = "detectiondjdin@gmail.comg"

    subject = f"URGENT: {alert_type} Detected!"
    body = f"""
    Alert: {alert_type} detected!
    Confidence Level: {confidence:.2f}%
    
    Immediate action is required.
    """

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Attach image
    with open(image_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)
        msg.add_attachment(file_data, maintype='image', subtype='png', filename=file_name)

    # Send email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"[EMAIL SENT] {alert_type} alert sent successfully!")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email: {e}")

# Loop over video frames
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean

    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    confidence = results[i]

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
        alert_text = f"Warning: {label}"
        cv2.putText(output, alert_text, (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 5)

        irand = randrange(0, 1000)
        filename = f"{irand}.png"
        image_path = os.path.join(img_dir, filename)
        cv2.imwrite(image_path, output)

        # Play alarm sound
        winsound.Beep(1000, 1000)

        # Announce alert
        announcement = f"{label} detected. Take action immediately."
        engine.say(announcement)
        engine.runAndWait()

        # Send Email Alert
        send_email(label, confidence * 100, image_path)

    output_resized = cv2.resize(output, (display_width, display_height))

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_dir, fourcc, 30, (W, H), True)
        if not writer.isOpened():
            print("[ERROR] Failed to initialize video writer.")
            break

    writer.write(output)
    cv2.imshow("Output", output_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] Cleaning up...")
if writer is not None:
    writer.release()
vs.release()
cv2.destroyAllWindows()
