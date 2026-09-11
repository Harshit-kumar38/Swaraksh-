import sys
import os
import yaml
import torch
import cv2
import numpy as np

# --------------------------------------------------
# DeepfakeBench paths
# --------------------------------------------------

sys.path.insert(0, os.path.abspath("."))

from detectors.xception_detector import XceptionDetector


CONFIG_PATH = "training/config/detector/xception.yaml"
WEIGHTS_PATH = "training/weights/xception_best.pth"

# Face image is outside DeepfakeBench
IMAGE_PATH = "../normalized_faces/face_0000.jpg"


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Use the ImageNet Xception backbone
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
# Build Xception detector
# --------------------------------------------------

print("Loading Xception model...")

model = XceptionDetector(config)


# --------------------------------------------------
# Load trained DeepfakeBench checkpoint
# --------------------------------------------------

print("Loading trained checkpoint...")

checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location="cpu"
)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print(
        "Checkpoint keys:",
        list(checkpoint.keys())[:20]
    )


# --------------------------------------------------
# Extract state dictionary
# --------------------------------------------------

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
# Remove DataParallel "module." prefix if present
# --------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


# --------------------------------------------------
# Load trained weights
# --------------------------------------------------

missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print()
    print("First missing keys:")
    print(missing[:10])

if unexpected:
    print()
    print("First unexpected keys:")
    print(unexpected[:10])


# --------------------------------------------------
# Evaluation mode
# --------------------------------------------------

model = model.to(device)
model.eval()


# --------------------------------------------------
# Load face image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )

print()
print("Input image:", IMAGE_PATH)
print("Original shape:", image.shape)


# --------------------------------------------------
# BGR -> RGB
# --------------------------------------------------

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# Resize to DeepfakeBench resolution
# --------------------------------------------------

image = cv2.resize(
    image,
    (256, 256),
    interpolation=cv2.INTER_AREA
)


# --------------------------------------------------
# Convert to float
# --------------------------------------------------

image = image.astype(
    np.float32
) / 255.0


# --------------------------------------------------
# Normalize
# mean = [0.5, 0.5, 0.5]
# std  = [0.5, 0.5, 0.5]
# --------------------------------------------------

mean = np.array(
    [0.5, 0.5, 0.5],
    dtype=np.float32
)

std = np.array(
    [0.5, 0.5, 0.5],
    dtype=np.float32
)

image = (image - mean) / std


# --------------------------------------------------
# HWC -> CHW
# --------------------------------------------------

image = np.transpose(
    image,
    (2, 0, 1)
)


# --------------------------------------------------
# Add batch dimension
# --------------------------------------------------

image = np.expand_dims(
    image,
    axis=0
)


# --------------------------------------------------
# Convert to PyTorch tensor
# --------------------------------------------------

tensor = torch.from_numpy(
    image
).float().to(device)


# --------------------------------------------------
# Run inference
# --------------------------------------------------

print()
print("Running inference...")

with torch.no_grad():

    data = {
        "image": tensor
    }

    output = model(data)

    probability = output["prob"].item()

    logits = (
        output["cls"]
        .detach()
        .cpu()
        .numpy()
    )


# --------------------------------------------------
# Display result
# --------------------------------------------------

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print("Logits:", logits)

print(
    f"Fake probability : {probability:.4f}"
)

print(
    f"Real probability : {1 - probability:.4f}"
)

if probability >= 0.5:
    print("Prediction       : FAKE")
else:
    print("Prediction       : REAL")

print("=" * 60)
