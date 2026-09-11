import onnx

INPUT = "models/aasist-l.onnx"
OUTPUT = "models/aasist-l-prepared.onnx"

print("Loading ONNX model...")
model = onnx.load(INPUT)

print(f"Original value_info entries: {len(model.graph.value_info)}")

# Remove intermediate value_info shape/type declarations.
# ONNX Runtime will infer what it needs during quantization.
model.graph.ClearField("value_info")

print("Cleared intermediate value_info entries.")

# Keep the actual model inputs and outputs untouched.
onnx.checker.check_model(model)

onnx.save(model, OUTPUT)

print("✅ Prepared model saved!")
print(f"Output: {OUTPUT}")
