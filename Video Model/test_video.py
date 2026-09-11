import cv2
import os


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration
    }


def extract_frames(video_path, output_folder="frames", sample_fps=2):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        raise ValueError("Invalid video FPS")

    # Number of original frames to skip
    interval = max(int(original_fps / sample_fps), 1)

    frame_number = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % interval == 0:

            filename = os.path.join(
                output_folder,
                f"frame_{saved_count:04d}.jpg"
            )

            cv2.imwrite(filename, frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    return saved_count


# -------------------------
# TEST
# -------------------------

video_path = "test.mp4"

info = get_video_info(video_path)

print("\nVIDEO INFORMATION")
print("------------------")
print("FPS:", info["fps"])
print("Total frames:", info["frame_count"])
print("Duration:", info["duration"], "seconds")

print("\nExtracting frames...")

count = extract_frames(
    video_path,
    output_folder="frames",
    sample_fps=2
)

print("Frames extracted:", count)