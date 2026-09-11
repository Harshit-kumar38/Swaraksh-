import asyncio
import websockets
import json
import base64
import numpy as np
import soundfile as sf
import g711
from scipy.signal import resample_poly


WS_URL = "ws://127.0.0.1:8000/stream"
AUDIO_FILE = "test_audio/elevenlabs.wav"


async def main():

    print("==========================================")
    print(" AASIST WEBSOCKET CODEC VALIDATION")
    print("==========================================")

    # --------------------------------------------------
    # Load original audio
    # --------------------------------------------------

    print(f"\nLoading: {AUDIO_FILE}")

    audio, sr = sf.read(AUDIO_FILE)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    print(f"Original sample rate: {sr}")
    print(f"Original samples: {len(audio)}")
    print(f"Original duration: {len(audio) / sr:.3f}s")

    if sr != 16000:
        raise ValueError("Expected 16 kHz audio")

    # --------------------------------------------------
    # 16 kHz → 8 kHz
    # --------------------------------------------------

    print("\nResampling 16 kHz → 8 kHz...")

    pcm8_float = resample_poly(
        audio,
        1,
        2
    ).astype(np.float32)

    print(f"8 kHz samples: {len(pcm8_float)}")

    # --------------------------------------------------
    # Convert float → int16
    # --------------------------------------------------

    pcm8_int16 = np.clip(
        pcm8_float * 32767,
        -32768,
        32767
    ).astype(np.int16)

    # --------------------------------------------------
    # int16 → μ-law
    # --------------------------------------------------

    print("Converting PCM → μ-law...")

    ulaw_bytes = g711.encode_ulaw(
        pcm8_int16.astype(np.float32) / 32768.0
    )

    print(f"μ-law bytes: {len(ulaw_bytes)}")

    # --------------------------------------------------
    # WebSocket
    # --------------------------------------------------

    print(f"\nConnecting to {WS_URL}")

    async with websockets.connect(
        WS_URL,
        max_size=None
    ) as websocket:

        print("✅ WebSocket connected")

        # --------------------------------------------------
        # START
        # --------------------------------------------------

        await websocket.send(
            json.dumps({
                "event": "start",
                "start": {
                    "callSid": "CODEC_TEST_001"
                }
            })
        )

        response = await websocket.recv()

        print(f"\nServer: {response}")

        # --------------------------------------------------
        # STREAM μ-law
        # --------------------------------------------------

        chunk_size = 160

        print("\n🎙️ Streaming μ-law audio...")
        print("20 ms chunks @ 8 kHz")

        total_bytes = 0
        chunk_count = 0

        for i in range(
            0,
            len(ulaw_bytes),
            chunk_size
        ):

            chunk = ulaw_bytes[
                i:i + chunk_size
            ]

            payload = base64.b64encode(
                chunk
            ).decode("utf-8")

            message = {
                "event": "media",
                "media": {
                    "payload": payload
                }
            }

            await websocket.send(
                json.dumps(message)
            )

            total_bytes += len(chunk)
            chunk_count += 1

            # Simulate telephony timing
            await asyncio.sleep(0.020)

        print("\n==========================================")
        print("STREAM COMPLETE")
        print("==========================================")

        print(
            f"μ-law bytes sent: {total_bytes}"
        )

        print(
            f"Audio chunks sent: {chunk_count}"
        )

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        await websocket.send(
            json.dumps({
                "event": "stop"
            })
        )

        # --------------------------------------------------
        # Wait for final server response
        # --------------------------------------------------

        print("\nWaiting for final analysis...")

        try:

            while True:

                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=5
                )

                print(
                    f"\n🔎 Server response:"
                )

                print(response)

                try:

                    data = json.loads(response)

                    if data.get("event") == "final_analysis":
                        break

                except:
                    pass

        except asyncio.TimeoutError:

            print(
                "\n⚠️ No final response received."
            )

    print("\n==========================================")
    print(" TEST COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    asyncio.run(main())