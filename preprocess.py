# ============================================================
#  preprocess.py  —  Fixed Data Collection Script
#  Collects hand landmark data for MULTIPLE sign classes
#  and saves to hand.csv for training
# ============================================================
#
#  HOW TO USE:
#  1. Run:  python preprocess.py
#  2. Choose N (new dataset) or C (continue adding more signs)
#  3. Type the sign word/phrase (e.g. "hello", "thank you", "good")
#  4. Show that hand sign in front of camera — hold steady
#  5. Press 'q' to stop recording that sign
#  6. Run again for EACH sign you want to add (choose C each time)
#  7. After collecting all signs → run training.py to retrain model
# ============================================================

import mediapipe as mp
import cv2
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ── Ask user: new file or continue ───────────────────────────────────────────
print("\n" + "="*50)
print("  SIGN LANGUAGE DATA COLLECTOR")
print("="*50)
data_choice = input("\nNew dataset (N) or Continue existing (C): ").strip().upper()
class_name  = input("Enter sign word or phrase (e.g. hello, good, thank you): ").strip()

if not class_name:
    print("No sign name entered. Exiting.")
    exit()

print(f"\nReady to record: '{class_name}'")
print("Show the sign in front of the camera.")
print("Hold the sign STEADY — data is captured every frame.")
print("Press 'q' when done.\n")

# ── CSV header ────────────────────────────────────────────────────────────────
# 21 landmarks × 4 values (x, y, z, visibility) = 84 columns + 1 class column
num_coords = 21
landmarks_header = ['class']
for val in range(1, num_coords + 1):
    landmarks_header += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

csv_file = 'hand.csv'

if data_choice == 'N':
    # Create fresh CSV with header
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(landmarks_header)
    print(f"Created new {csv_file}")
else:
    # Append to existing — create with header if file doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(landmarks_header)
        print(f"Created new {csv_file} (file didn't exist)")
    else:
        print(f"Appending to existing {csv_file}")

# ── Webcam capture ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
frame_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10),  thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

        # ── Try left hand first, then right ──────────────────────────────
        hand_landmarks = None
        if results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks.landmark
        elif results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark

        detected = False
        if hand_landmarks:
            pose_row = list(
                np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in hand_landmarks
                ]).flatten()
            )
            row = [class_name] + pose_row

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            frame_count += 1
            detected = True

        # ── Overlay ───────────────────────────────────────────────────────
        color = (0, 255, 0) if detected else (0, 0, 255)
        status = f"Recording: {class_name}  |  Frames: {frame_count}" if detected else "NO HAND DETECTED — show your hand!"
        cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(image, status, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'q' when done", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(f"Collecting: {class_name}", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Done! Recorded {frame_count} frames for '{class_name}'")
print(f"   Saved to {csv_file}")

# Show current class counts
import pandas as pd
try:
    df = pd.read_csv(csv_file)
    print("\nCurrent dataset summary:")
    print(df['class'].value_counts().to_string())
    print(f"\nTotal rows: {len(df)}")
    print("\nNext step: run  python training.py  to retrain the model")
except Exception as e:
    print(f"Could not read summary: {e}")
