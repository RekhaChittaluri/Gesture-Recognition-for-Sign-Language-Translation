import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure 'model.p' is in the current directory.")
except KeyError:
    raise Exception("Model key not found in the pickle file. Ensure the file contains a dictionary with a 'model' key.")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Ensure the correct camera index is used
if not cap.isOpened():
    raise Exception("Could not open video device")

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

# Dictionary for label interpretation
labels_dict = {i: chr(65 + i) for i in range(26)}  # Labels for A-Z

while True:
    data_aux = []

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        continue

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_ = []
            y_ = []
            num_landmarks = len(hand_landmarks.landmark)  # Number of landmarks for one hand

            for i in range(num_landmarks):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Calculate normalized coordinates
            min_x = min(x_)
            min_y = min(y_)
            for i in range(num_landmarks):
                norm_x = x_[i] - min_x
                norm_y = y_[i] - min_y
                data_aux.append(norm_x)
                data_aux.append(norm_y)

            # Ensure data_aux has exactly 42 features (21 landmarks * 2 coordinates)
            if len(data_aux) > 42:
                data_aux = data_aux[:42]
            elif len(data_aux) < 42:
                print("Insufficient landmarks detected.")
                continue

            # Convert to numpy array and reshape for prediction
            data_aux = np.asarray(data_aux).reshape(1, -1)

            # Perform prediction
            prediction = model.predict(data_aux)

            predicted_character = labels_dict[int(prediction[0])]

            # Display predicted character on the frame
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()