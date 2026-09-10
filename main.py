"""Run ONNX lane segmentation on a video using NVIDIA CUDA."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DATA_DIR = Path("/data")
MODEL_FILENAME = "deeplabv3plus_resnet50.onnx"
IMG_SIZE = (640, 640)
ALPHA = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment lane pixels in a video with ONNX Runtime CUDA."
    )
    parser.add_argument("input_video", help="Input video filename in the project folder")
    parser.add_argument("output_video", help="Output video filename in the project folder")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def filename_in_data_dir(filename: str, label: str) -> Path:
    """Allow filenames only, keeping all video I/O inside the mounted project folder."""
    path = Path(filename)
    if path.name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"{label} must be a filename, not a path: {filename}")
    return DATA_DIR / path


def create_mask(frame: np.ndarray, session: ort.InferenceSession, input_name: str,
                output_name: str, size: tuple[int, int]) -> np.ndarray:
    image = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # tensorflow.keras.applications.resnet50.preprocess_input equivalent
    image = image[..., ::-1]
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    image = np.expand_dims(image, axis=0)

    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and input_shape[1] == 3:
        image = np.transpose(image, (0, 3, 1, 2))

    prediction = session.run([output_name], {input_name: image})[0][0]
    if prediction.ndim == 3:
        if prediction.shape[-1] <= 10:  # HWC class logits
            mask = np.argmax(prediction, axis=-1)
        elif prediction.shape[0] <= 10:  # CHW class logits
            mask = np.argmax(prediction, axis=0)
        else:
            raise RuntimeError(f"Unexpected model output shape: {prediction.shape}")
    elif prediction.ndim == 2:
        mask = (prediction > 0.5).astype(np.uint8) if np.issubdtype(
            prediction.dtype, np.floating
        ) else prediction
    else:
        raise RuntimeError(f"Unexpected model output shape: {prediction.shape}")
    return mask.astype(np.uint8)


def overlay_lane(frame: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy()
    overlay[mask == 1] = [0, 255, 0]
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def main() -> int:
    args = parse_args()
    model_path = DATA_DIR / MODEL_FILENAME
    input_path = filename_in_data_dir(args.input_video, "Input video")
    output_path = filename_in_data_dir(args.output_video, "Output video")
    require_file(model_path, "ONNX model")
    require_file(input_path, "Input video")

    providers = ort.get_available_providers()
    print(f"ONNX Runtime {ort.__version__}; available providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Start Docker with --gpus all and install NVIDIA Container Toolkit."
        )

    session = ort.InferenceSession(
        str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Model did not load on CUDA: {session.get_providers()}")
    print(f"Active providers: {session.get_providers()}")
    model_input = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    started = time.monotonic()
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            mask = create_mask(
                frame, session, model_input.name, output_name, IMG_SIZE
            )
            writer.write(overlay_lane(frame, mask, ALPHA))
            count += 1
            if count % 100 == 0:
                elapsed = time.monotonic() - started
                print(f"Processed {count}/{total} frames ({count / elapsed:.2f} FPS)")
    finally:
        capture.release()
        writer.release()

    elapsed = time.monotonic() - started
    print(f"Complete: {count} frames in {elapsed:.2f}s ({count / elapsed:.2f} FPS)")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
