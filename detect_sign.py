import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


class SignLanguageDetector:
    def __init__(self, model_path="models/sign_language_model.pkl"):
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Load the trained model
        self.model = None
        self.labels_map = None
        self.load_model(model_path)

        # For prediction smoothing
        self.prediction_history = []
        self.history_size = 5

    def load_model(self, model_path):
        """
        Load the trained model and labels mapping
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please run train_model.py first to train a model."
            )

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.labels_map = model_data["labels_map"]

        print(f"Model loaded from {model_path}")
        print(f"Available labels: {sorted(self.labels_map.values())}")

    def normalize_landmarks(self, landmarks):
        """
        Normalize hand landmarks relative to the wrist position
        and scale to make them translation and scale invariant
        """
        # Convert landmarks to numpy array
        coords = []
        for landmark in landmarks:
            coords.extend([landmark.x, landmark.y, landmark.z])

        coords = np.array(coords).reshape(-1, 3)

        # Get wrist position (first landmark)
        wrist = coords[0]

        # Translate to origin (wrist at 0,0,0)
        coords = coords - wrist

        # Calculate max distance for normalization
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords = coords / max_dist

        # Flatten back to 1D array
        return coords.flatten()

    def get_prediction(self, landmarks):
        """
        Get prediction for the given hand landmarks
        """
        # Normalize landmarks
        normalized_landmarks = self.normalize_landmarks(landmarks)

        # Reshape for prediction
        features = normalized_landmarks.reshape(1, -1)

        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Get the predicted label
        predicted_label = self.labels_map[prediction]
        confidence = probabilities[prediction]

        return predicted_label, confidence

    def smooth_prediction(self, prediction):
        """
        Smooth predictions using a sliding window
        """
        self.prediction_history.append(prediction)

        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        # Return most common prediction
        if len(self.prediction_history) >= 3:
            return max(set(self.prediction_history), key=self.prediction_history.count)
        return prediction

    def detect_realtime(self):
        """
        Run real-time sign language detection
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 50)
        print("REAL-TIME SIGN LANGUAGE DETECTION")
        print("=" * 50)
        print("Press 'q' to quit")
        print("Press 'c' to clear prediction history")
        print("=" * 50 + "\n")

        current_prediction = None
        current_confidence = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = self.hands.process(rgb_frame)

            # Draw hand landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Get prediction
                    try:
                        predicted_label, confidence = self.get_prediction(
                            hand_landmarks.landmark
                        )

                        # Smooth prediction
                        smoothed_prediction = self.smooth_prediction(predicted_label)

                        current_prediction = smoothed_prediction
                        current_confidence = confidence

                    except Exception as e:
                        print(f"Prediction error: {e}")
                        current_prediction = "Error"
                        current_confidence = 0.0
            else:
                # No hand detected, clear history
                self.prediction_history = []
                current_prediction = None

            # Display UI elements
            self.draw_ui(frame, current_prediction, current_confidence)

            # Show frame
            cv2.imshow("Sign Language Detection", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nExiting...")
                break
            elif key == ord("c"):
                self.prediction_history = []
                print("Prediction history cleared")

        cap.release()
        cv2.destroyAllWindows()

    def draw_ui(self, frame, prediction, confidence):
        """
        Draw UI elements on the frame
        """
        h, w, _ = frame.shape

        # Draw semi-transparent overlay for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw title
        cv2.putText(
            frame,
            "Sign Language Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Draw prediction
        if prediction is not None:
            # Main prediction display
            prediction_text = f"Sign: {prediction}"
            confidence_text = f"Confidence: {confidence * 100:.1f}%"

            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            cv2.putText(
                frame,
                prediction_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )

            cv2.putText(
                frame,
                confidence_text,
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Draw large letter in top-right corner
            letter_size = 3
            letter_thickness = 5
            text_size = cv2.getTextSize(
                prediction, cv2.FONT_HERSHEY_SIMPLEX, letter_size, letter_thickness
            )[0]
            text_x = w - text_size[0] - 20
            text_y = text_size[1] + 20

            # Draw shadow for better visibility
            cv2.putText(
                frame,
                prediction,
                (text_x + 3, text_y + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                letter_size,
                (0, 0, 0),
                letter_thickness + 2,
            )

            cv2.putText(
                frame,
                prediction,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                letter_size,
                color,
                letter_thickness,
            )

        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Draw instructions at bottom
        cv2.putText(
            frame,
            "Press 'q' to quit | 'c' to clear",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def main():
    try:
        # Initialize detector
        detector = SignLanguageDetector(model_path="models/sign_language_model.pkl")

        # Start real-time detection
        detector.detect_realtime()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease follow these steps:")
        print("1. Run 'python collect_data.py' to collect training data")
        print("2. Run 'python train_model.py' to train the model")
        print("3. Run 'python detect_sign.py' to start detection")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
