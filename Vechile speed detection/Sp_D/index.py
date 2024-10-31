import warnings
# Ignore warnings from the google.protobuf module
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import cv2  # OpenCV for image processing
import mediapipe as mp  # MediaPipe for hand tracking
import numpy as np  # NumPy for numerical operations
import os  # For directory operations
from tensorflow.keras.models import Sequential, load_model  # Keras for model building
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Layers for LSTM model
from tensorflow.keras.utils import to_categorical  # For one-hot encoding
import argparse  # For command-line argument parsing

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1) # Initialize hands with a maximum of 1 hand

DATA_PATH = 'gesture_data'  # Path to save the collected data
sequence_length = 30        # Sequence length for the LSTM model
actions = ['HELLO', '1', 'A', '3', 'V']  # Define gestures/signs

def extract_keypoints(hand_landmarks):
    """Extracts x, y, z coordinates from hand landmarks."""
    return [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

def collect_data():
    """Collects gesture data from the webcam and saves it to files."""
    cap = cv2.VideoCapture(0)
    
    while True:
        # Continuously prompt the user to enter a gesture action to collect data for,
        # converting the input to uppercase to match the defined actions.
        action = input("Enter the gesture action to collect (e.g., 'A', 'B', 'C') or type 'exit' to stop: ").upper()
        
        if action == 'EXIT':
            break
        
        if action not in actions:
            print("Invalid action. Please enter a valid action or 'exit' to stop.")
            continue
        
        action_dir = os.path.join(DATA_PATH, action) # Directory for the current action
        os.makedirs(action_dir, exist_ok=True) # Create directory if it doesn't exist
        print(f"Collecting data for action '{action}'")

        for sequence in range(30):  # Collect 30 sequences per action
            frame_sequence = [] # Store keypoints for this sequence
            print(f"Collecting sequence {sequence + 1}/30 for action '{action}'. Press 'q' to stop early.")
            
            for frame_num in range(sequence_length):
                ret, frame = cap.read() # Read a frame from the webcam
                if not ret:
                    print("Failed to grab frame. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB
                result = hands.process(frame_rgb) # Process frame to find hands

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Draw hand landmarks on the frame
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        keypoints = extract_keypoints(hand_landmarks) # Extract keypoints
                        frame_sequence.append(keypoints) # Add keypoints to the sequence
                
                cv2.imshow('Collecting Data', frame) # Show the collecting data window
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Data collection stopped by user.")
                    break

            # Pad with zeros if necessary
            while len(frame_sequence) < sequence_length:
                frame_sequence.append(np.zeros((21, 3))) # Add zero vectors to maintain length

            np.save(os.path.join(action_dir, f'seq_{sequence}.npy'), frame_sequence) # Save the sequence as .npy file
        
        print(f"Completed collecting data for action: {action}.")
    
    cap.release() # Release the webcam
    cv2.destroyAllWindows() # Close all OpenCV windows

def train_model():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)} # Map actions to numbers

    for action in actions:
        for sequence_file in os.listdir(os.path.join(DATA_PATH, action)):
            sequence = np.load(os.path.join(DATA_PATH, action, sequence_file)) # Load sequence data
            sequences.append(sequence) # Add to sequences list
            labels.append(label_map[action]) # Add corresponding label

    X = np.array(sequences).reshape((len(sequences), sequence_length, 63)) # Reshape for LSTM
    y = to_categorical(labels).astype(int) # One-hot encode labels


 # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dropout(0.5)) # Regularization to prevent overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(actions), activation='softmax')) # Output layer for multi-class classification

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile model
    model.fit(X, y, epochs=500, batch_size=32, validation_split=0.2) # Train model

    model.save('action.h5') # Save the trained model
    print("Model trained and saved as 'action.h5'")

def real_time_recognition():
    """Runs real-time gesture recognition using the trained model."""
    
    cap = cv2.VideoCapture(0) # Start video capture from the webcam
    model = load_model('action.h5') # Load the trained model
    sequence = [] # Initialize sequence for predictions
    threshold = 0.8 # Confidence threshold for predictions

    while cap.isOpened():
        ret, frame = cap.read() # Read a frame
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
        result = hands.process(frame_rgb) # Process frame

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = extract_keypoints(hand_landmarks) # Extract keypoints
                sequence.append(keypoints) # Add keypoints to the sequence

                if len(sequence) == sequence_length: # If enough frames have been collected
                    sequence_np = np.array(sequence).reshape(1, sequence_length, 21 * 3) # Reshape for model input
                    prediction = model.predict(sequence_np)[0] # Predict gesture
                    predicted_class = np.argmax(prediction) # Get predicted class index

                    if prediction[predicted_class] > threshold: # Check if prediction meets threshold
                        accuracy = prediction[predicted_class] * 100  # Convert to percentage
                        
                        # Display predicted text and accuracy on the frame
                        cv2.putText(frame, f"Predicted Text: {actions[predicted_class]}", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (184, 134, 11), 2)

                    sequence.pop(0)  # Remove the oldest sequence to maintain the correct length

        cv2.imshow('Real-Time Gesture Recognition', frame) # Show the frame with predictions

        if cv2.waitKey(10) & 0xFF == ord('q'): # Exit on 'q' key press
            break

    cap.release() # Release the webcam
    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help="Collect new gesture data")
    parser.add_argument('--train', action='store_true', help="Train LSTM model on collected data")
    parser.add_argument('--recognize', action='store_true', help="Run real-time gesture recognition")
    args = parser.parse_args()


# Execute the corresponding function based on the argument provided
    if args.collect:
        collect_data()
    elif args.train:
        train_model()
    elif args.recognize:
        real_time_recognition()
    else:
        print("Please specify an action: --collect, --train, or --recognize")