from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT = "models/aasist-l-prepared.onnx"
OUTPUT = "models/aasist-l-int8.onnx"

print("Quantizing AASIST-L to INT8...")

quantize_dynamic(
    model_input=INPUT,
    model_output=OUTPUT,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False,
)

print("✅ Quantization complete!")
print(f"Output: {OUTPUT}")