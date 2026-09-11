from ultralytics import YOLO
import cv2
import glob
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "yolov11n-face.pt"
INPUT_DIR = "lovi_frames"
OUTPUT_DIR = "lovi_face_frames"

CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 1280

# Minimum acceptable face size
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading face detection model...")

model = YOLO(MODEL_PATH)

frames = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
)

if not frames:
    raise RuntimeError(
        f"No frames found in: {INPUT_DIR}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Frames found:", len(frames))
print("-" * 50)


# --------------------------------------------------
# Process frames
# --------------------------------------------------

face_number = 0
frames_with_faces = 0
rejected_faces = 0

for frame_path in frames:

    image = cv2.imread(frame_path)

    if image is None:
        print(
            "Could not read:",
            frame_path
        )
        continue

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    frame_face_count = 0

    for box in result.boxes:

        # Get bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Keep coordinates inside image
        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop face
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_height, face_width = face.shape[:2]

        # --------------------------------------------------
        # Face quality filter
        # --------------------------------------------------

        if (
            face_width < MIN_FACE_WIDTH
            or face_height < MIN_FACE_HEIGHT
        ):
            rejected_faces += 1

            print(
                f"{os.path.basename(frame_path)} -> "
                f"Rejected small face: "
                f"{face_width}x{face_height}"
            )

            continue

        # --------------------------------------------------
        # Save valid face
        # --------------------------------------------------

        filename = f"face_{face_number:04d}.jpg"

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            face
        )

        face_number += 1
        frame_face_count += 1

        print(
            f"{os.path.basename(frame_path)} -> "
            f"{filename} "
            f"({face_width}x{face_height})"
        )

    if frame_face_count > 0:
        frames_with_faces += 1


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "-" * 50)

print("Frames processed:", len(frames))
print("Frames with valid faces:", frames_with_faces)
print("Valid face crops:", face_number)
print("Rejected small faces:", rejected_faces)
print("Output folder:", OUTPUT_DIR)

print("-" * 50)
