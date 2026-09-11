import torch
import os
import sys

sys.path.append("./models")

from AASIST import Model as AASISTNet

print("Loading AASIST-L model...")

# EXACT AASIST-L configuration
d_args = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# Create model with correct configuration
model = AASISTNet(d_args)

# Load checkpoint
checkpoint = torch.load(
    "models/AASIST-L.pth",
    map_location="cpu"
)

# Your checkpoint is already an OrderedDict
state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

model.eval()

print("✅ Model loaded successfully!")

# Dummy input
dummy_input = torch.randn(1, 64600)

print("Exporting to ONNX...")

# AASIST returns TWO outputs:
# last_hidden, output
torch.onnx.export(

    model,

    dummy_input,

    "models/aasist-l.onnx",

    opset_version=18,

    input_names=["input"],

    output_names=["embedding", "output"],

    dynamo=True,

    verbose=False,

)

print("✅ Model exported successfully!")

size_mb = os.path.getsize("models/aasist-l.onnx") / 1024 / 1024
print(f"File size: {size_mb:.2f} MB")