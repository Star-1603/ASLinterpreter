import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

# Define the letters corresponding to each class
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define a region of interest (ROI) where the hand will be placed
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI (resize, normalize, etc.)
    roi_resized = cv2.resize(roi, (64, 64))
    roi_resized = roi_resized.astype('float32') / 255.0
    roi_resized = np.expand_dims(roi_resized, axis=0)

    # Make prediction
    prediction = model.predict(roi_resized)
    predicted_class = np.argmax(prediction)
    predicted_letter = classes[predicted_class]

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {predicted_letter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw rectangle around the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show the video stream with predictions
    cv2.imshow('Sign Language Interpreter', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
