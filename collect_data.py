import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time

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


def draw_hand_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(lm.x * width), int(lm.y * height))
        for lm in landmarks
    ]

    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def normalize_landmarks(landmarks):
        coords = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
            dtype=np.float32,
        )
        coords -= coords[0]  # put wrist at origin
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords /= max_dist
        return coords.flatten()

    def collect_samples(self, label, num_samples=100):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open the webcam. Check that it is connected and not being used by another app.")
            return []

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        samples = []
        collecting = False
        timestamp_ms = 0
        last_sample_time = 0.0

        print(f"\n=== Collecting data for letter: {label} ===")
        print(f"Press SPACE to start collecting {num_samples} samples")
        print("Press 'q' to quit this letter")

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
                    has_hand = bool(result.hand_landmarks)

                    if has_hand:
                        hand = result.hand_landmarks[0]
                        draw_hand_landmarks(frame, hand)

                        now = time.monotonic()
                        if collecting and len(samples) < num_samples and now - last_sample_time >= 0.05:
                            samples.append(self.normalize_landmarks(hand))
                            last_sample_time = now

                    cv2.putText(
                        frame,
                        f"Label: {label} | Samples: {len(samples)}/{num_samples}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    status = "COLLECTING..." if collecting else "Press SPACE to start"
                    cv2.putText(
                        frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2,
                    )

                    if not has_hand:
                        cv2.putText(
                            frame, "No hand detected!", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )

                    cv2.imshow("Data Collection", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("\nQuitting collection for this letter.")
                        break
                    if key == ord(" ") and not collecting:
                        collecting = True
                        print("Started collecting...")

                    if len(samples) >= num_samples:
                        print(f"Collected {len(samples)} samples for '{label}'.")
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return samples

    def save_data(self, data, labels, filename="sign_language_data.pkl"):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "wb") as file:
            pickle.dump({"data": data, "labels": labels}, file)
        print(f"\nData saved to {filepath}")
        print(f"Total samples: {len(data)}")

    def load_existing_data(self, filename="sign_language_data.pkl"):
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as file:
                data_dict = pickle.load(file)
            return data_dict["data"], data_dict["labels"]
        return [], []

    def collect_alphabet_data(self, alphabets, samples_per_letter=100):
        all_data, all_labels = self.load_existing_data()

        print("=" * 50)
        print("SIGN LANGUAGE DATA COLLECTION")
        print("=" * 50)
        print(f"\nCollecting data for: {', '.join(alphabets)}")
        print(f"Samples per letter: {samples_per_letter}")

        for alphabet in alphabets:
            response = input(
                f"\nReady to collect data for '{alphabet}'? (y/n/skip): "
            ).lower()

            if response in ("skip", "s"):
                print(f"Skipping '{alphabet}'")
                continue
            if response != "y":
                print("Stopping data collection.")
                break

            samples = self.collect_samples(alphabet, samples_per_letter)
            if samples:
                all_data.extend(samples)
                all_labels.extend([alphabet] * len(samples))
                self.save_data(all_data, all_labels)
                print(f"Collected {len(samples)} samples for '{alphabet}'.")
            else:
                print(f"No samples collected for '{alphabet}'.")

        print("\n" + "=" * 50)
        print("DATA COLLECTION COMPLETE")
        print("=" * 50)
        print(f"Total samples: {len(all_data)}")
        print(f"Total labels: {len(set(all_labels))}")
        return all_data, all_labels


def main():
    print("\nWELCOME TO SIGN LANGUAGE DATA COLLECTOR")
    print("-" * 50)
    print("Show the sign, then press SPACE to collect samples.")
    print("Use good lighting and keep the hand visible.")
    print("-" * 50)

    alphabets = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    choice = input("\nCollect all alphabets A-Z? (y/n): ").lower()

    if choice != "y":
        custom_input = input("Enter letters (e.g., A B C): ").upper()
        alphabets = custom_input.split()

    try:
        samples_input = input("Samples per letter (default 100): ")
        samples_per_letter = int(samples_input) if samples_input else 100
    except ValueError:
        samples_per_letter = 100

    collector = DataCollector()
    collector.collect_alphabet_data(alphabets, samples_per_letter)


if __name__ == "__main__":
    main()