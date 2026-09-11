from fastapi import FastAPI, WebSocket
from fastapi.responses import Response

import numpy as np
import base64
import scipy.signal
import g711
import time

from aasist_m4_inference import AASISTM4Inference


# ==========================================================
# APP
# ==========================================================

app = FastAPI(
    title="Real-Time Voice Deepfake Detection",
    version="1.0.0"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "models/aasist-l.onnx"

SAMPLE_RATE = 16000

WINDOW_SAMPLES = 64600
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE

SPOOF_THRESHOLD = 0.85

# Keep last 2 seconds after each analysis
OVERLAP_SAMPLES = 32000


# ==========================================================
# NGROK CONFIGURATION
# ==========================================================

NGROK_URL = "vaseline-identity-resonate.ngrok-free.dev"


# ==========================================================
# LOAD MODEL ONCE
# ==========================================================

print("\n==========================================")
print(" Loading AASIST-L Detection Engine")
print("==========================================")

engine = AASISTM4Inference(
    onnx_path=MODEL_PATH,
    use_coreml=True
)

print("Model loaded successfully")
print("==========================================\n")


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "Real-Time Voice Deepfake Detection",
        "model": "AASIST-L",
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": round(WINDOW_SECONDS, 2),
        "spoof_threshold": SPOOF_THRESHOLD
    }


# ==========================================================
# TWILIO VOICE WEBHOOK
# ==========================================================

@app.post("/twiml")
@app.get("/twiml")
async def twilio_webhook():

    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>

    <Start>
        <Stream
            name="voice-deepfake-detection"
            url="wss://vaseline-identity-resonate.ngrok-free.dev/stream"
        />
    </Start>

    <Say>
        Your call is protected by real-time voice security.
    </Say>

    <Pause length="60"/>

</Response>
"""

    print("📞 Twilio called /twiml")

    return Response(
        content=twiml,
        media_type="application/xml"
    )

# ==========================================================
# μ-LAW 8kHz → PCM FLOAT32 16kHz
# ==========================================================

def ulaw_to_pcm16k(ulaw_bytes: bytes) -> np.ndarray:

    # ------------------------------------------------------
    # Decode G.711 μ-law
    # ------------------------------------------------------

    pcm_8k = g711.decode_ulaw(
        ulaw_bytes
    )

    pcm_8k = np.asarray(
        pcm_8k,
        dtype=np.float32
    )

    # ------------------------------------------------------
    # 8kHz → 16kHz
    # ------------------------------------------------------

    pcm_16k = scipy.signal.resample_poly(
        pcm_8k,
        up=2,
        down=1
    )

    return pcm_16k.astype(
        np.float32
    )


# ==========================================================
# TWILIO MEDIA STREAM
# ==========================================================

@app.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print("\n==========================================")
    print("🎙️ TWILIO CLIENT CONNECTED")
    print("==========================================")

    audio_buffer = []

    call_sid = "unknown"
    stream_sid = "unknown"

    analysis_count = 0

    stream_start_time = time.time()

    try:

        while True:

            # ==================================================
            # RECEIVE TWILIO MESSAGE
            # ==================================================

            data = await websocket.receive_json()

            event = data.get("event")

            print(
                f"📡 Twilio event: {event}"
            )


            # ==================================================
            # CONNECTED EVENT
            # ==================================================

            if event == "connected":

                print(
                    "✅ Twilio WebSocket connected"
                )


            # ==================================================
            # START EVENT
            # ==================================================

            elif event == "start":

                start_data = data.get(
                    "start",
                    {}
                )

                call_sid = start_data.get(
                    "callSid",
                    "unknown"
                )

                stream_sid = start_data.get(
                    "streamSid",
                    data.get(
                        "streamSid",
                        "unknown"
                    )
                )

                media_format = start_data.get(
                    "mediaFormat",
                    {}
                )

                print("\n==========================================")
                print("📞 CALL STARTED")
                print("==========================================")

                print(
                    f"Call SID: {call_sid}"
                )

                print(
                    f"Stream SID: {stream_sid}"
                )

                print(
                    f"Encoding: "
                    f"{media_format.get('encoding')}"
                )

                print(
                    f"Sample Rate: "
                    f"{media_format.get('sampleRate')}"
                )

                print(
                    f"Channels: "
                    f"{media_format.get('channels')}"
                )

                print(
                    "🟢 Deepfake detection active"
                )


            # ==================================================
            # MEDIA EVENT
            # ==================================================

            elif event == "media":

                media = data.get(
                    "media",
                    {}
                )

                payload = media.get(
                    "payload"
                )

                if not payload:
                    continue


                # ------------------------------------------------
                # Base64 decode
                # ------------------------------------------------

                try:

                    ulaw_bytes = base64.b64decode(
                        payload
                    )

                except Exception as e:

                    print(
                        f"❌ Base64 decode error: {e}"
                    )

                    continue


                # ------------------------------------------------
                # μ-law 8kHz → 16kHz
                # ------------------------------------------------

                pcm_16k = ulaw_to_pcm16k(
                    ulaw_bytes
                )


                # ------------------------------------------------
                # Add audio to rolling buffer
                # ------------------------------------------------

                audio_buffer.extend(
                    pcm_16k.tolist()
                )


                # ------------------------------------------------
                # Display progress approximately every second
                # ------------------------------------------------

                if (
                    len(audio_buffer) >= SAMPLE_RATE
                    and
                    len(audio_buffer) % SAMPLE_RATE
                    < len(pcm_16k)
                ):

                    print(
                        f"📊 Buffer: "
                        f"{len(audio_buffer)} / "
                        f"{WINDOW_SAMPLES} "
                        f"("
                        f"{len(audio_buffer) / SAMPLE_RATE:.2f}s"
                        f")"
                    )


                # ==================================================
                # RUN AASIST WHEN WINDOW IS READY
                # ==================================================

                while len(audio_buffer) >= WINDOW_SAMPLES:

                    analysis_count += 1

                    print("\n==========================================")

                    print(
                        f"🔍 AASIST ANALYSIS #{analysis_count}"
                    )

                    print("==========================================")


                    # ------------------------------------------------
                    # Take exactly 4.04 seconds
                    # ------------------------------------------------

                    audio_chunk = np.asarray(
                        audio_buffer[:WINDOW_SAMPLES],
                        dtype=np.float32
                    )


                    # ------------------------------------------------
                    # Run inference
                    # ------------------------------------------------

                    inference_start = time.perf_counter()

                    spoof_score = engine.predict(
                        audio_chunk
                    )

                    inference_time = (
                        time.perf_counter()
                        - inference_start
                    ) * 1000


                    spoof_score = float(
                        spoof_score
                    )

                    bona_fide_score = float(
                        1.0 - spoof_score
                    )


                    # ------------------------------------------------
                    # Decision
                    # ------------------------------------------------

                    is_spoof = (
                        spoof_score >= SPOOF_THRESHOLD
                    )


                    # ==================================================
                    # DISPLAY RESULT
                    # ==================================================

                    print(
                        f"🎭 Spoof Score: "
                        f"{spoof_score:.4f}"
                    )

                    print(
                        f"🎤 Real/Bona-fide Score: "
                        f"{bona_fide_score:.4f}"
                    )

                    print(
                        f"⚡ Inference Time: "
                        f"{inference_time:.2f} ms"
                    )


                    if is_spoof:

                        print(
                            "\n🚨🚨🚨"
                        )

                        print(
                            "🚨 LIKELY DEEPFAKE VOICE 🚨"
                        )

                        print(
                            "🚨🚨🚨"
                        )

                    else:

                        print(
                            "\n✅ LIKELY REAL VOICE"
                        )


                    # ==================================================
                    # SEND RESULT
                    # ==================================================

                    await websocket.send_json({

                        "event": "analysis",

                        "analysis_number":
                            analysis_count,

                        "callSid":
                            call_sid,

                        "spoof_score":
                            spoof_score,

                        "bona_fide_score":
                            bona_fide_score,

                        "is_spoof":
                            is_spoof,

                        "result":
                            (
                                "LIKELY DEEPFAKE"
                                if is_spoof
                                else "LIKELY REAL"
                            ),

                        "threshold":
                            SPOOF_THRESHOLD,

                        "latency_ms":
                            inference_time

                    })


                    # ==================================================
                    # DEEPFAKE DETECTED
                    # ==================================================

                    if is_spoof:

                        print("\n==========================================")
                        print("🚨 SECURITY ALERT")
                        print("==========================================")

                        print(
                            f"📞 Call SID: {call_sid}"
                        )

                        print(
                            f"🎭 Spoof Score: "
                            f"{spoof_score:.4f}"
                        )

                        print(
                            f"⚠️ Threshold: "
                            f"{SPOOF_THRESHOLD:.2f}"
                        )

                        print(
                            "🚨 DEEPFAKE VOICE DETECTED"
                        )

                        print(
                            "=========================================="
                        )


                        await websocket.send_json({

                            "event":
                                "spoof_detected",

                            "callSid":
                                call_sid,

                            "score":
                                spoof_score,

                            "bona_fide_score":
                                bona_fide_score,

                            "action":
                                "alert"

                        })


                    # ------------------------------------------------
                    # Keep last 2 seconds
                    # ------------------------------------------------

                    audio_buffer = (
                        audio_buffer[
                            -OVERLAP_SAMPLES:
                        ]
                    )


            # ==================================================
            # STOP EVENT
            # ==================================================

            elif event == "stop":

                duration = (
                    time.time()
                    - stream_start_time
                )

                print("\n==========================================")
                print("🛑 CALL / STREAM STOPPED")
                print("==========================================")

                print(
                    f"📞 Call SID: {call_sid}"
                )

                print(
                    f"⏱️ Duration: "
                    f"{duration:.2f}s"
                )

                print(
                    f"📊 Analyses performed: "
                    f"{analysis_count}"
                )

                print(
                    "=========================================="
                )

                break


            # ==================================================
            # UNKNOWN EVENT
            # ==================================================

            else:

                print(
                    f"⚠️ Unknown event: {event}"
                )


    except Exception as e:

        print(
            f"\n❌ WebSocket error: {e}"
        )


    finally:

        try:

            await websocket.close()

        except Exception:

            pass

        print(
            "\n🔌 WebSocket connection closed"
        )


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    print(
        "\n🚀 Starting Voice Deepfake Detection..."
    )

    print(
        "=========================================="
    )

    print(
        "HTTP: "
        "http://127.0.0.1:8000"
    )

    print(
        "Twilio Webhook: "
        f"https://{NGROK_URL}/twiml"
    )

    print(
        "Twilio WebSocket: "
        f"wss://{NGROK_URL}/stream"
    )

    print(
        f"Threshold: "
        f"{SPOOF_THRESHOLD}"
    )

    print(
        "=========================================="
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )