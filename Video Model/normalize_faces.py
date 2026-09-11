import cv2
import glob
import os

input_dir = "face_frames"
output_dir = "normalized_faces"

os.makedirs(output_dir, exist_ok=True)

faces = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

print("Total face crops:", len(faces))

processed = 0

for face_path in faces:

    image = cv2.imread(face_path)

    if image is None:
        print("Could not read:", face_path)
        continue

    # Resize to 224 x 224
    face = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    filename = os.path.basename(face_path)

    output_path = os.path.join(
        output_dir,
        filename
    )

    cv2.imwrite(output_path, face)

    processed += 1

print("-" * 50)
print("Normalized faces:", processed)
print("Output folder:", output_dir)