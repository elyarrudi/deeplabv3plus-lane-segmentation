# ONNX Runtime 1.23.x is built for CUDA 12.x and cuDNN 9.x.
# The NVIDIA Container Toolkit passes the host GPU/driver into this image at runtime.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

COPY main.py export_onnx.py /app/

# Example:
# Export once (GPU is not required):
# docker run --rm -v "${PWD}:/data" --entrypoint python3 lane-segment /app/export_onnx.py
#
# Run inference (GPU required):
# docker run --rm --gpus all -v "${PWD}:/data" lane-segment input.mp4 output.mp4
ENTRYPOINT ["python3", "/app/main.py"]
CMD ["--help"]
