import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = os.path.join("models", "hand_landmarker.task")
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def create_hand_landmarker(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {model_path}\n"
            "Download hand_landmarker.task into the models folder first."
        )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


class SignLanguageDetector:
    def __init__(self, model_path="models/sign_language_model.pkl"):
        self.model = None
        self.labels_map = None
        self.load_model(model_path)
        self.prediction_history = []
        self.history_size = 5

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Collect samples and run train_model.py first."
            )

        with open(model_path, "rb") as file:
            model_data = pickle.load(file)

        self.model = model_data["model"]
        self.labels_map = model_data["labels_map"]
        print(f"Classifier loaded from {model_path}")
        print(f"Available labels: {sorted(self.labels_map.values())}")

    @staticmethod
    def normalize_landmarks(landmarks):
        coords = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
            dtype=np.float32,
        )
        coords -= coords[0]
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords /= max_dist
        return coords.flatten()

    def get_prediction(self, landmarks):
        features = self.normalize_landmarks(landmarks).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        label = self.labels_map[prediction]
        confidence = float(probabilities[prediction])
        return label, confidence

    def smooth_prediction(self, prediction):
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        if len(self.prediction_history) >= 3:
            return max(set(self.prediction_history), key=self.prediction_history.count)
        return prediction

    def detect_realtime(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check that it is connected and not being used by another app."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nREAL-TIME SIGN LANGUAGE DETECTION")
        print("Press 'q' to quit | 'c' to clear prediction history")

        timestamp_ms = 0
        current_prediction = None
        current_confidence = 0.0

        try:
            with create_hand_landmarker() as landmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    timestamp_ms += 1
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.hand_landmarks:
                        hand = result.hand_landmarks[0]
                        draw_hand_landmarks(frame, hand)
                        try:
                            label, confidence = self.get_prediction(hand)
                            current_prediction = self.smooth_prediction(label)
                            current_confidence = confidence
                        except Exception as error:
                            print(f"Prediction error: {error}")
                            current_prediction = "Error"
                            current_confidence = 0.0
                    else:
                        self.prediction_history = []
                        current_prediction = None
                        current_confidence = 0.0

                    self.draw_ui(frame, current_prediction, current_confidence)
                    cv2.imshow("Sign Language Detection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\nExiting.")
                        break
                    if key == ord("c"):
                        self.prediction_history = []
                        print("Prediction history cleared.")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def draw_ui(self, frame, prediction, confidence):
        height, width, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.putText(
            frame, "Sign Language Detection", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        if prediction is not None:
            text_color = (
                (0, 255, 0) if confidence > 0.8
                else (0, 255, 255) if confidence > 0.6
                else (0, 165, 255)
            )
            cv2.putText(
                frame, f"Sign: {prediction}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3,
            )
            cv2.putText(
                frame, f"Confidence: {confidence * 100:.1f}%", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            letter_size = 3
            thickness = 5
            text_size = cv2.getTextSize(
                prediction, cv2.FONT_HERSHEY_SIMPLEX, letter_size, thickness
            )[0]
            x = width - text_size[0] - 20
            y = text_size[1] + 20
            cv2.putText(
                frame, prediction, (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                letter_size, (0, 0, 0), thickness + 2,
            )
            cv2.putText(
                frame, prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                letter_size, text_color, thickness,
            )
        else:
            cv2.putText(
                frame, "No hand detected", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        cv2.putText(
            frame, "Press 'q' to quit | 'c' to clear",
            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1,
        )


def main():
    try:
        detector = SignLanguageDetector(
            model_path="models/sign_language_model.pkl"
        )
        detector.detect_realtime()
    except FileNotFoundError as error:
        print(f"\nError: {error}")
        print("Collect samples, train the model, then run this detector.")
    except Exception as error:
        print(f"\nAn error occurred: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()