from ultralytics import YOLO
import cv2
import glob
import os

model = YOLO("yolov11n-face.pt")

frames = sorted(glob.glob("frames/*.jpg"))

output_dir = "face_frames"
os.makedirs(output_dir, exist_ok=True)

face_number = 0

for frame_path in frames:

    image = cv2.imread(frame_path)

    if image is None:
        continue

    results = model.predict(
        source=image,
        conf=0.5,
        imgsz=1280,
        verbose=False
    )

    result = results[0]

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Make sure coordinates stay inside image
        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop face
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        filename = f"face_{face_number:04d}.jpg"

        cv2.imwrite(
            os.path.join(output_dir, filename),
            face
        )

        face_number += 1

        print(
            f"{os.path.basename(frame_path)} -> {filename}"
        )

print("\n" + "-" * 50)
print("Total face crops:", face_number)
print("Saved in:", output_dir)
