import onnxruntime as ort
import numpy as np
import soundfile as sf
import time
import sys


class AASISTM4Inference:

    def __init__(
        self,
        onnx_path="models/aasist-l.onnx",
        use_coreml=True
    ):

        print(f"Loading model: {onnx_path}")

        # --------------------------------------------------
        # Check available ONNX Runtime execution providers
        # --------------------------------------------------

        available_providers = ort.get_available_providers()

        print("\nAvailable providers:")
        for provider in available_providers:
            print(f"  - {provider}")

        # --------------------------------------------------
        # Select CoreML if available
        # Otherwise use CPU
        # --------------------------------------------------

        providers = []

        if (
            use_coreml
            and "CoreMLExecutionProvider"
            in available_providers
        ):
            providers.append(
                "CoreMLExecutionProvider"
            )

        # CPU fallback
        providers.append(
            "CPUExecutionProvider"
        )

        print("\nRequested providers:")
        print(providers)

        # --------------------------------------------------
        # Create ONNX Runtime session
        # --------------------------------------------------

        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )

        print("\nActual execution providers:")
        print(
            self.session.get_providers()
        )

        # --------------------------------------------------
        # AASIST-L input requirements
        # --------------------------------------------------

        self.sample_rate = 16000

        # 4.04 seconds × 16000 Hz
        self.window_samples = 64600

        # --------------------------------------------------
        # Print model information
        # --------------------------------------------------

        print("\nModel input:")

        for inp in self.session.get_inputs():

            print(
                f"  {inp.name}: "
                f"{inp.shape} "
                f"{inp.type}"
            )

        print("\nModel outputs:")

        for output in self.session.get_outputs():

            print(
                f"  {output.name}: "
                f"{output.shape} "
                f"{output.type}"
            )

        print("\n✅ AASIST-L M4 inference engine ready!")

    # ======================================================
    # PREPARE AUDIO
    # ======================================================

    def prepare_audio(
        self,
        pcm_audio: np.ndarray
    ):

        # Make sure audio is float32
        pcm_audio = pcm_audio.astype(
            np.float32
        )

        # --------------------------------------------------
        # Check empty audio
        # --------------------------------------------------

        if len(pcm_audio) == 0:

            raise ValueError(
                "Audio input is empty."
            )

        # --------------------------------------------------
        # Pad short audio
        # --------------------------------------------------

        if len(pcm_audio) < self.window_samples:

            repeats = int(
                np.ceil(
                    self.window_samples
                    / len(pcm_audio)
                )
            )

            pcm_audio = np.tile(
                pcm_audio,
                repeats
            )

            pcm_audio = pcm_audio[
                :self.window_samples
            ]

        # --------------------------------------------------
        # Crop long audio
        # --------------------------------------------------

        elif len(pcm_audio) > self.window_samples:

            pcm_audio = pcm_audio[
                :self.window_samples
            ]

        # --------------------------------------------------
        # Final shape
        # --------------------------------------------------

        pcm_audio = pcm_audio.astype(
            np.float32
        )

        return pcm_audio

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        pcm_audio: np.ndarray
    ) -> float:

        # Prepare exactly 64600 samples
        pcm_audio = self.prepare_audio(
            pcm_audio
        )

        # ONNX input shape:
        # [batch, samples]
        wav_tensor = pcm_audio.reshape(
            1,
            self.window_samples
        )

        # --------------------------------------------------
        # Run inference
        # --------------------------------------------------

        start = time.perf_counter()

        outputs = self.session.run(
            None,
            {
                "input": wav_tensor
            }
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # --------------------------------------------------
        # IMPORTANT:
        #
        # outputs[0] = embedding [1,160]
        # outputs[1] = classifier [1,2]
        # --------------------------------------------------

        embedding = outputs[0]

        logits = outputs[1][0]

        # --------------------------------------------------
        # Stable softmax
        # --------------------------------------------------

        logits = (
            logits
            - np.max(logits)
        )

        exp_logits = np.exp(
            logits
        )

        probabilities = (
            exp_logits
            / np.sum(exp_logits)
        )

        # --------------------------------------------------
        # AASIST-L class order
        #
        # index 0 = spoof
        # index 1 = bona fide
        # --------------------------------------------------

        spoof_prob = float(
            probabilities[0]
        )

        bona_fide_prob = float(
            probabilities[1]
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if spoof_prob < 0.50:

            result = "LIKELY REAL"

        elif spoof_prob < 0.80:

            result = "SUSPICIOUS"

        else:

            result = "LIKELY DEEPFAKE"

        # --------------------------------------------------
        # Print result
        # --------------------------------------------------

        print(
            f"\nInference latency: "
            f"{latency_ms:.2f} ms"
        )

        print(
            f"Bona fide probability: "
            f"{bona_fide_prob:.4f}"
        )

        print(
            f"Spoof probability: "
            f"{spoof_prob:.4f}"
        )

        print(
            f"Embedding shape: "
            f"{embedding.shape}"
        )

        print(
            f"Result: {result}"
        )

        return spoof_prob


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        " AASIST-L M4 INFERENCE TEST"
    )

    print(
        "=========================================="
    )

    # Create inference engine
    engine = AASISTM4Inference(
        onnx_path="models/aasist-l.onnx",
        use_coreml=True
    )

    # ------------------------------------------------------
    # Create dummy 4.04-second audio
    # ------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python3 aasist_m4_inference.py "
            "test_audio/real.wav"
        )

        sys.exit(1)


    audio_path = sys.argv[1]

    print(
        f"\nLoading audio: {audio_path}"
    )

    audio, sr = sf.read(
        audio_path
    )

    print(
        f"Original sample rate: {sr}"
    )

    print(
        f"Original shape: {audio.shape}"
    )

    # Stereo → mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    # Make float32
    audio = audio.astype(
        np.float32
    )

    # Check sample rate
    if sr != engine.sample_rate:

        print(
            f"ERROR: Expected "
            f"{engine.sample_rate} Hz "
            f"but received {sr} Hz."
        )

        sys.exit(1)

    print(
        f"Audio samples: {len(audio)}"
    )

    print(
        f"Duration: "
        f"{len(audio) / engine.sample_rate:.2f} seconds"
    )

    print(
        "\nRunning audio inference..."
    )

    score = engine.predict(
        audio
    )

    print(
        "\n=========================================="
    )

    print(
        f"Final spoof score: {score:.4f}"
    )

    print(
        "=========================================="
    )

    