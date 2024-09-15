import cv2
import numpy as np
import tensorflow as tf

# Load the trained model (Ensure the path is correct)
try:
    model = tf.keras.models.load_model('asl_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the letters corresponding to each class
classes = list("A B C D del E F G H I J K L M N nothing O P Q R S space T U V W X Y Z")

# Capture video from webcam (use 0 for default webcam, change if using an external camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a consistent frame size for the video (optional, based on your webcam resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # Define a region of interest (ROI) where the hand will be placed
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Display the ROI for reference (optional)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Preprocess the ROI (resize, normalize, etc.)
    roi_resized = cv2.resize(roi, (64, 64))  # Assuming the model input is (64, 64, 3)
    roi_resized = roi_resized.astype('float32') / 255.0  # Normalize pixel values
    roi_resized = np.expand_dims(roi_resized, axis=0)  # Add batch dimension (1, 64, 64, 3)

    # Try to make a prediction (use try-except to handle potential errors)
    try:
        prediction = model.predict(roi_resized)
        predicted_class = np.argmax(prediction)
        predicted_letter = classes[predicted_class]

        # Display the prediction on the frame
        cv2.putText(frame, f'Prediction: {predicted_letter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Prediction error: {e}")

    # Show the video stream with predictions
    cv2.imshow('Sign Language Interpreter', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
