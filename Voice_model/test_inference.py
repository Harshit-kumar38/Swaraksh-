import onnxruntime as ort
import numpy as np
import time


class AASISTInference:
    def __init__(self, model_path="models/aasist-l-int8.onnx"):
        print(f"Loading model from {model_path}...")

        # Start with CPU for INT8 ONNX inference.
        providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=ort.SessionOptions()
        )

        self.sample_rate = 16000
        self.window_samples = 64600

        print("✅ Model loaded!")
        print(f"Expected input: {self.window_samples} samples @ {self.sample_rate}Hz")
        print(f"Execution provider: {self.session.get_providers()}")

        # Print model I/O information
        print("\nModel inputs:")
        for inp in self.session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")

        print("\nModel outputs:")
        for out in self.session.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")

    def predict(self, audio: np.ndarray) -> float:
        """
        audio:
            float32 mono PCM audio at 16 kHz.

        Returns:
            spoof probability
            0.0 = likely bona fide
            1.0 = likely spoof/deepfake
        """

        # Make sure audio is float32
        audio = np.asarray(audio, dtype=np.float32)

        # Ensure exactly 64600 samples
        if len(audio) < self.window_samples:
            audio = np.pad(
                audio,
                (0, self.window_samples - len(audio))
            )
        elif len(audio) > self.window_samples:
            audio = audio[-self.window_samples:]

        # Shape: [batch, samples]
        input_tensor = audio.reshape(1, -1)

        # Run inference
        start_time = time.perf_counter()

        outputs = self.session.run(
            None,
            {"input": input_tensor}
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # outputs[0] = embedding
        # outputs[1] = classification logits
        embedding = outputs[0]
        logits = outputs[1]

        # Convert logits to probabilities
        logits = logits[0]

        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        bona_fide_prob = float(probabilities[0])
        spoof_prob = float(probabilities[1])

        print(
            f"⏱️ Latency: {latency_ms:.2f} ms | "
            f"🎭 Spoof score: {spoof_prob:.4f} | "
            f"Real: {bona_fide_prob:.4f}"
        )

        print(f"Embedding shape: {embedding.shape}")

        return spoof_prob


if __name__ == "__main__":

    print("🚀 Starting AASIST-L INT8 inference test...\n")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    engine = AASISTInference()

    # ---------------------------------------------------------
    # Test 1: Random noise
    # ---------------------------------------------------------

    print("\n📊 Test 1: Random noise")

    noise = (
        np.random.randn(64600)
        .astype(np.float32)
        * 0.1
    )

    noise_score = engine.predict(noise)

    # ---------------------------------------------------------
    # Test 2: 440 Hz sine wave
    # ---------------------------------------------------------

    print("\n📊 Test 2: 440 Hz sine wave")

    t = np.arange(64600, dtype=np.float32) / 16000.0

    sine_wave = (
        np.sin(2 * np.pi * 440 * t)
        .astype(np.float32)
        * 0.5
    )

    sine_score = engine.predict(sine_wave)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 50)
    print("INFERENCE TEST COMPLETE")
    print("=" * 50)

    print(f"Random noise spoof score : {noise_score:.4f}")
    print(f"Sine wave spoof score    : {sine_score:.4f}")

    print("\nInterpretation:")
    print("0.00 - 0.50 → likely bona fide")
    print("0.50 - 0.80 → uncertain / suspicious")
    print("0.80 - 1.00 → likely spoof")

    print("\n⚠️ Important:")
    print("Random noise and sine waves are NOT valid tests")
    print("for real-world deepfake detection.")
    print("Use real speech and known synthetic speech for evaluation.")
