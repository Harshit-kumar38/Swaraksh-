from ultralytics import YOLO
import glob
import os

# Load face detector
model = YOLO("yolov11n-face.pt")

# Get sampled frames
frames = sorted(glob.glob("frames/*.jpg"))

print(f"Total frames: {len(frames)}")
print("-" * 50)

total_faces = 0
frames_with_faces = 0

for frame_path in frames:
    results = model.predict(
        source=frame_path,
        conf=0.5,
        verbose=False
    )

    face_count = len(results[0].boxes)

    if face_count > 0:
        frames_with_faces += 1
        total_faces += face_count

    print(
        f"{os.path.basename(frame_path)} "
        f"→ faces: {face_count}"
    )

print("-" * 50)
print(f"Frames with faces: {frames_with_faces}/{len(frames)}")
print(f"Total detected faces: {total_faces}")