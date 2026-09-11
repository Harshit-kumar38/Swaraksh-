import cv2
import os

VIDEO_PATH = "lovi.mp4"
OUTPUT_DIR = "lovi_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total frames:", total_frames)

# Sample approximately 2 frames per second
interval = max(1, int(fps / 2))

frame_index = 0
saved = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % interval == 0:

        output_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved:04d}.jpg"
        )

        cv2.imwrite(output_path, frame)

        saved += 1

    frame_index += 1

cap.release()

print("-" * 50)
print("Frames saved:", saved)
print("Output folder:", OUTPUT_DIR)
