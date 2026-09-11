import sys
import os
import yaml
import torch
import cv2
import numpy as np
import glob

# --------------------------------------------------
# DeepfakeBench paths
# --------------------------------------------------

sys.path.insert(0, os.path.abspath("."))

from detectors.xception_detector import XceptionDetector


CONFIG_PATH = "training/config/detector/xception.yaml"
WEIGHTS_PATH = "training/weights/xception_best.pth"
FACE_DIR = "../normalized_faces"


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

config["pretrained"] = "training/pretrained/xception-b5690688.pth"


# --------------------------------------------------
# Select device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 60)
print("DEVICE:", device)
print("=" * 60)


# --------------------------------------------------
# Build model
# --------------------------------------------------

print("Loading Xception model...")

model = XceptionDetector(config)


# --------------------------------------------------
# Load trained checkpoint
# --------------------------------------------------

print("Loading trained checkpoint...")

checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# --------------------------------------------------
# Remove module. prefix
# --------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


# --------------------------------------------------
# Load weights
# --------------------------------------------------

missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print("Missing:", missing[:10])

if unexpected:
    print("Unexpected:", unexpected[:10])


# --------------------------------------------------
# Evaluation mode
# --------------------------------------------------

model = model.to(device)
model.eval()


# --------------------------------------------------
# Find all face images
# --------------------------------------------------

face_paths = sorted(
    glob.glob(
        os.path.join(FACE_DIR, "*.jpg")
    )
)

print()
print("Total face crops:", len(face_paths))

if len(face_paths) == 0:
    raise RuntimeError(
        "No face images found in " + FACE_DIR
    )


# --------------------------------------------------
# Run inference on every face
# --------------------------------------------------

probabilities = []

print()
print("=" * 60)
print("RUNNING XCEPTION ON ALL FACE CROPS")
print("=" * 60)

for index, face_path in enumerate(face_paths):

    image = cv2.imread(face_path)

    if image is None:
        print(
            f"[{index + 1}/{len(face_paths)}] "
            f"Could not read {face_path}"
        )
        continue

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to DeepfakeBench resolution
    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float
    image = image.astype(
        np.float32
    ) / 255.0

    # Normalize
    mean = np.array(
        [0.5, 0.5, 0.5],
        dtype=np.float32
    )

    std = np.array(
        [0.5, 0.5, 0.5],
        dtype=np.float32
    )

    image = (image - mean) / std

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    tensor = torch.from_numpy(
        image
    ).float().to(device)

    # Inference
    with torch.no_grad():

        data = {
            "image": tensor
        }

        output = model(data)

        probability = output["prob"].item()

    probabilities.append(probability)

    print(
        f"[{index + 1:02d}/{len(face_paths)}] "
        f"{os.path.basename(face_path)} "
        f"→ fake probability: {probability:.4f}"
    )


# --------------------------------------------------
# Video-level aggregation
# --------------------------------------------------

probabilities = np.array(
    probabilities,
    dtype=np.float32
)

mean_probability = float(
    np.mean(probabilities)
)

median_probability = float(
    np.median(probabilities)
)

minimum_probability = float(
    np.min(probabilities)
)

maximum_probability = float(
    np.max(probabilities)
)

fake_faces = int(
    np.sum(probabilities >= 0.5)
)

real_faces = int(
    np.sum(probabilities < 0.5)
)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=" * 60)
print("VIDEO-LEVEL XCEPTION RESULT")
print("=" * 60)

print(
    f"Faces analyzed       : {len(probabilities)}"
)

print(
    f"Average fake prob.   : {mean_probability:.4f}"
)

print(
    f"Median fake prob.    : {median_probability:.4f}"
)

print(
    f"Minimum fake prob.   : {minimum_probability:.4f}"
)

print(
    f"Maximum fake prob.   : {maximum_probability:.4f}"
)

print(
    f"Faces classified fake: {fake_faces}"
)

print(
    f"Faces classified real: {real_faces}"
)

print()

if mean_probability >= 0.5:
    print("VIDEO PREDICTION     : FAKE")
else:
    print("VIDEO PREDICTION     : REAL")

print("=" * 60)
