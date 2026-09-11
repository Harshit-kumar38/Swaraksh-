from ultralytics import YOLO
import cv2
import glob
import os

model = YOLO("yolov11n-face.pt")

frames = sorted(glob.glob("frames/*.jpg"))

output_dir = "face_check"
os.makedirs(output_dir, exist_ok=True)

for frame_path in frames:

    image = cv2.imread(frame_path)

    results = model.predict(
        source=image,
        conf=0.5,
        imgsz=1280,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes
    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Face {confidence:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    filename = os.path.basename(frame_path)

    cv2.imwrite(
        os.path.join(output_dir, filename),
        image
    )

    print(
        f"{filename} -> {len(result.boxes)} faces"
    )

print("\nDone.")
print("Check the 'face_check' folder.")