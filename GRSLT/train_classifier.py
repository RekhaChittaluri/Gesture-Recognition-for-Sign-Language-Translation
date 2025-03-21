import pickle
import cv2
import mediapipe as mp
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Test the trained model with video capture
cap = cv2.VideoCapture(0)  # Ensure the correct camera index is used
if not cap.isOpened():
    raise Exception("Could not open video device")

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary for label interpretation
labels_dict = {i: chr(65 + i) for i in range(26)}

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
        num_landmarks = len(results.multi_hand_landmarks[0].landmark)  # Number of landmarks for one hand

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(num_landmarks):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

        # Calculate normalized coordinates
        for i in range(num_landmarks):
            norm_x = x_[i] - min(x_)
            norm_y = y_[i] - min(y_)
            data_aux.append(norm_x)
            data_aux.append(norm_y)

        # Ensure data_aux has exactly 42 features (if more, take first 42)
        data_aux = data_aux[:42]

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