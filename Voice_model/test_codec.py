import numpy as np
import soundfile as sf
import scipy.signal
import g711

from aasist_m4_inference import AASISTM4Inference


AUDIO_FILE = "test_audio/asvspoof/known_real.flac"

SAMPLE_RATE = 16000


print("\n==========================================")
print(" AASIST G.711 CODEC ROUND-TRIP TEST")
print("==========================================")


# ==========================================================
# LOAD ORIGINAL AUDIO
# ==========================================================

print(f"\nLoading: {AUDIO_FILE}")

audio, sr = sf.read(AUDIO_FILE)

print(f"Original sample rate: {sr}")
print(f"Original samples: {len(audio)}")


# Stereo → mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)


audio = audio.astype(np.float32)


if sr != 16000:
    raise ValueError(
        f"Expected 16000 Hz, got {sr}"
    )


# ==========================================================
# ORIGINAL AUDIO
# ==========================================================

print("\n------------------------------------------")
print("1. ORIGINAL AUDIO")
print("------------------------------------------")

engine = AASISTM4Inference(
    onnx_path="models/aasist-l.onnx",
    use_coreml=True
)

original_score = engine.predict(audio)

print(
    f"\nOriginal spoof score: "
    f"{original_score:.4f}"
)


# ==========================================================
# 16 kHz → 8 kHz
# ==========================================================

print("\n------------------------------------------")
print("2. 16 kHz → 8 kHz")
print("------------------------------------------")

pcm8 = scipy.signal.resample_poly(
    audio,
    up=1,
    down=2
).astype(np.float32)

print(
    f"8 kHz samples: {len(pcm8)}"
)


# ==========================================================
# 8 kHz PCM → μ-law
# ==========================================================

print("\n------------------------------------------")
print("3. PCM → μ-law")
print("------------------------------------------")

ulaw = g711.encode_ulaw(
    pcm8
)

print(
    f"μ-law bytes: {len(ulaw)}"
)


# ==========================================================
# μ-law → 8 kHz PCM
# ==========================================================

print("\n------------------------------------------")
print("4. μ-law → PCM")
print("------------------------------------------")

decoded_pcm8 = g711.decode_ulaw(
    ulaw
)

decoded_pcm8 = np.asarray(
    decoded_pcm8,
    dtype=np.float32
)

print(
    f"Decoded 8 kHz samples: "
    f"{len(decoded_pcm8)}"
)


# ==========================================================
# 8 kHz → 16 kHz
# ==========================================================

print("\n------------------------------------------")
print("5. 8 kHz → 16 kHz")
print("------------------------------------------")

decoded_pcm16 = scipy.signal.resample_poly(
    decoded_pcm8,
    up=2,
    down=1
).astype(np.float32)

print(
    f"Decoded 16 kHz samples: "
    f"{len(decoded_pcm16)}"
)


# ==========================================================
# AASIST AFTER CODEC
# ==========================================================

print("\n------------------------------------------")
print("6. AASIST AFTER G.711 ROUND TRIP")
print("------------------------------------------")

codec_score = engine.predict(
    decoded_pcm16
)


print("\n==========================================")
print(" CODEC TEST RESULT")
print("==========================================")

print(
    f"Original spoof score: "
    f"{original_score:.4f}"
)

print(
    f"After G.711 score:    "
    f"{codec_score:.4f}"
)

print("==========================================")


if original_score < 0.5 and codec_score < 0.5:

    print(
        "\n✅ CODEC PIPELINE PASSED"
    )

elif original_score < 0.5 and codec_score >= 0.5:

    print(
        "\n❌ CODEC PIPELINE FAILED"
    )

    print(
        "\nThe audio becomes incorrectly classified "
        "after G.711 conversion."
    )

else:

    print(
        "\n⚠️ Original audio itself is being "
        "classified as spoof."
    )