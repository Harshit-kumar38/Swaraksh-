import sys
import os
import glob
import cv2
import torch
import numpy as np

# Add DeepfakeBench to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.detectors.xception_detector import XceptionDetector


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)

# Load configuration
config = {
    "pretrained": "./training/pretrained/xception-b5690688.pth",
    "model_name": "xception",
    "backbone_name": "xception",
    "backbone_config": {
        "mode": "original",
        "num_classes": 2,
        "inc": 3,
        "dropout": False
    },
    "loss_func": "cross_entropy"
}

print("Loading Xception model...")

model = XceptionDetector(config)
model = model.to(DEVICE)
model.eval()

print("Loading trained checkpoint...")

checkpoint = torch.load(
    "./training/weights/xception_best.pth",
    map_location=DEVICE
)

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

print("Checkpoint loaded successfully.")

faces = sorted(
    glob.glob("../lovi_normalized_faces/*.jpg")
)

print("Faces found:", len(faces))
print("-" * 50)

probabilities = []

for i, face_path in enumerate(faces, 1):

    image = cv2.imread(face_path)

    if image is None:
        print("Could not read:", face_path)
        continue

    # BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Xception expects 256x256
    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float
    image = image.astype(np.float32) / 255.0

    # Normalize using DeepfakeBench settings
    image = (image - 0.5) / 0.5

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model({
            "image": tensor
        })

    fake_probability = output["prob"].item()

    probabilities.append(fake_probability)

    print(
        f"[{i:02d}/{len(faces)}] "
        f"{os.path.basename(face_path)} "
        f"→ {fake_probability:.4f}"
    )


print("-" * 50)

if probabilities:

    probabilities = np.array(probabilities)

    print("Faces analyzed:", len(probabilities))
    print(f"Average fake probability : {probabilities.mean():.4f}")
    print(f"Median fake probability  : {np.median(probabilities):.4f}")
    print(f"Minimum fake probability  : {probabilities.min():.4f}")
    print(f"Maximum fake probability  : {probabilities.max():.4f}")

    fake_count = np.sum(probabilities >= 0.5)

    print(f"Faces >= 0.5              : {fake_count}")
    print(f"Faces < 0.5               : {len(probabilities) - fake_count}")

    video_probability = probabilities.mean()

    if video_probability >= 0.5:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    print("-" * 50)
    print(f"Video fake probability    : {video_probability:.4f}")
    print(f"Video prediction          : {prediction}")

else:
    print("No faces available for analysis.")
