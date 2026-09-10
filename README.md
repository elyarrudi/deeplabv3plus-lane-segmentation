# 🚗 Lane Segmentation with DeepLabV3+

> **Pixel-accurate lane segmentation for video, from TensorFlow training to GPU-accelerated ONNX inference.** ✨

This project implements a **custom-built DeepLabV3+ with a ResNet50 backbone and output stride 16**. The DeepLabV3+ extension, ASPP, decoder, and output head are implemented manually in TensorFlow/Keras on top of an ImageNet-pretrained ResNet50 backbone. The trained network is then exported to ONNX and served through a GPU-accelerated ONNX Runtime video pipeline.

## 📥 Model downloads

The project uses two model files, each needed for a different step. Download them from Google Drive:

| Artifact | Format | Used by | Download |
| --- | --- | --- | --- |
| `bestmodel.keras` | Keras / TensorFlow | `export_onnx.py` (to build the ONNX file) | [Google Drive](https://drive.google.com/file/d/1cm036kcJNxYr6nICo83tS9Grb5batw-V/view?usp=drive_link) |
| `deeplabv3plus_resnet50.onnx` | ONNX (opset 17) | `main.py` (video inference) | [Google Drive](https://drive.google.com/file/d/1latOJ9iuqIE43wdEtRXDhdJP2BLKj7kq/view?usp=drive_link) |

> 🔗 The ONNX file is a format-converted copy of the Keras checkpoint — both come from the same training run. Download the `.keras` file only if you plan to re-export or re-evaluate the model; download the `.onnx` file if you only want to run inference.

## 🔥 Highlights

- **Hand-built DeepLabV3+ head** — encoder extension, ASPP, decoder, and output head are assembled layer by layer, not imported from a library.
- **Custom 3-block `conv5` extension** with dilated 3×3 convolutions (rate 2) to enlarge the receptive field.
- **ASPP** with rates 6, 12, 18 plus a global-pooling branch for multi-scale lane context.
- **Composite loss for class imbalance**: `Focal (α=0.25, γ=2.0) + 0.5 × weighted Dice`.
- **Frozen-backbone training** with a custom segmentation head — best validation Mean IoU of **0.7969**.
- **Strict CUDA inference** — `main.py` refuses to fall back to CPU if `CUDAExecutionProvider` is unavailable.
- **NHWC / NCHW auto-detection** in the ONNX runner.

> 🔎 **Note on originality:** the DeepLabV3+ architecture and training pipeline are custom, manually-written implementations. Only the ResNet50 backbone uses ImageNet-pretrained weights.

## 🌟 Results preview

<p align="center">
  <img src="1.png" width="32%" alt="Lane segmentation result 1" />
  <img src="2.png" width="32%" alt="Lane segmentation result 2" />
  <img src="3.png" width="32%" alt="Lane segmentation result 3" />
</p>

🎬 Three sample output videos are also included in the project root so the model's video-level results can be viewed directly.

## 🧠 Training

The complete training workflow is in [`train.ipynb`](train.ipynb). Training uses **one unified dataset created by combining two separate lane-segmentation datasets**. Their frames and binary lane masks were consolidated into one standardized `frames` / `lane-masks` structure before validation and splitting.

🔒 The two source datasets were downloaded from online sources, but their original download links are no longer available to the project author. If the unified dataset is required for reproduction or research, it can be provided separately on request.

The notebook validates image/mask pairs and performs a **sequence-grouped train/validation split** (`random_state=42`). Grouping frames that belong to the same video sequence keeps near-duplicate frames on one side of the split, avoiding leakage between the training and validation sets.

### Data pipeline and augmentation 🛠️

- Input resolution: **640 × 640 × 3**; batch size: **8**; output classes: **background** and **lane**.
- Images are preprocessed with the ImageNet-compatible **ResNet50 preprocessing** (caffe mode: channel reversal followed by mean subtraction).
- Geometric augmentation applies synchronized left/right flipping, random crop, and resize to both the image and its mask.
- Photometric augmentation adds random brightness, contrast, saturation, hue, and Gaussian-noise variation to the image.
- A cutout-style patch augmentation is applied with 30 % probability, injecting a random noise patch into the image to improve robustness against occlusions.
- `tf.data` uses parallel mapping, shuffling, batching, and prefetching for efficient GPU feeding. ⚡

### Custom DeepLabV3+ architecture 🔬

The following architecture description is taken directly from the model-building code in [`train.ipynb`](train.ipynb). It is a carefully customized **DeepLabV3+ implementation with output stride 16**, built manually in TensorFlow/Keras on top of the pretrained ResNet50 backbone.

| Component | Design |
| --- | --- |
| Encoder | ImageNet-pretrained **ResNet50** through `conv4_block6_out` |
| Custom encoder extension | Three manually implemented residual `conv5` blocks appended after `conv4_block6_out`; each uses 1×1 → dilated 3×3 → 1×1 convolutions, Batch Normalization, ReLU, and a residual shortcut. The first block applies a 1×1 projection on the shortcut to reach 2048 channels; the remaining two use identity shortcuts. The 3×3 branch uses dilation rate **2** |
| Context module | **ASPP** with global average pooling plus 1×1 and 3×3 atrous branches at dilation rates **6, 12, and 18** |
| Decoder | Bilinear upsampling, a 48-channel low-level skip connection from `conv2_block3_out`, concatenation, and refinement convolutions |
| Regularization | Dropout of **0.20** after ASPP and **0.10** in the decoder |
| Output | A 1×1, two-channel logits head; argmax / softmax is applied during loss and metric computation |

### Loss, optimizer, and safeguards 🎯

Lane pixels occupy far less area than background, so plain pixel accuracy is not sufficient. Training therefore uses a composite objective:

```text
Combined loss = Focal loss (α = 0.25, γ = 2.0) + 0.5 × weighted Dice loss
```

- **Focal loss** emphasizes difficult and minority-class pixels.
- **Weighted Dice loss** directly rewards mask overlap and improves sensitivity to thin lane markings.
- **AdamW** is used with an initial learning rate of `2e-4` and weight decay of `1e-4`.
- The ResNet backbone is frozen during the selected Stage-1 training run, allowing the custom segmentation head to adapt without immediately disturbing pretrained visual features.
- `ModelCheckpoint` saves the weights with the highest validation Mean IoU; `EarlyStopping` restores the best weights; `ReduceLROnPlateau` halves the learning rate when validation loss stops improving. 🏆

### Training strategy and results 📊

The ResNet50 backbone is kept frozen during the selected Stage-1 training run. This preserves the pretrained visual features and lets the new head adapt without disturbing them. The best checkpoint was reached at **epoch 14** and achieved the highest observed validation Mean IoU, so it was saved as the deployment candidate.

A selective fine-tuning stage (unfreezing the last 11 backbone layers at a lower learning rate) is also present in `train.ipynb`, but it was **not used as part of the final deployment model**. The reason is a clear generalization signal from Stage 1: after epoch 14, training metrics kept improving while `val_loss` began to rise and `val_mean_iou` no longer exceeded 0.7969. Continuing to fine-tune the pretrained backbone under this widening train/validation gap would have increased the risk of overfitting without a measurable gain in validation performance. The epoch-14 checkpoint was therefore retained as the final model.

Selected epoch-14 metrics:

| Metric | Training | Validation |
| --- | ---: | ---: |
| Loss | 0.052005 | 0.082236 |
| Pixel accuracy | 0.9887 | 0.9790 |
| Dice score | 0.9080 | 0.8650 |
| Mean IoU | 0.8574 | **0.7969** |
| Learning rate | `1.00e-04` | `1.00e-04` |

🔎 Later epochs continued to improve training-side metrics but did not exceed this validation IoU; validation loss also rose. This widening train/validation gap is a classic sign that further fitting could reduce generalization. For that reason, the epoch-14 checkpoint was retained.

## 📦 Keras → ONNX export

[`export_onnx.py`](export_onnx.py) loads `bestmodel.keras`, exports it with **tf2onnx** at **opset 17**, validates it with `onnx.checker`, and writes:

```text
deeplabv3plus_resnet50.onnx
```

This step converts `bestmodel.keras` into `deeplabv3plus_resnet50.onnx`. Only the `.keras` file is needed here — the `.onnx` file is its output.

The exported model expects input in **NHWC** format: `[batch, 640, 640, 3]`.

## ⚙️ Inference with `main.py`

[`main.py`](main.py) is the production video runner. It:

1. Loads the ONNX model with ONNX Runtime's `CUDAExecutionProvider`.
2. Reads the input video frame by frame.
3. Resizes and applies the same ResNet50 preprocessing used at training time.
4. Handles either NHWC or NCHW ONNX inputs automatically.
5. Converts class logits to a binary lane mask, overlays lane pixels in green, and writes the output video. ✅

## 🐳 Docker quick start

### Prerequisites

- Docker Engine (Linux) or Docker Desktop with the WSL 2 backend (Windows).
- An NVIDIA GPU, compatible NVIDIA driver, and NVIDIA Container Toolkit / Docker GPU support for `main.py`.

Place `bestmodel.keras` in the project root to export ONNX. For video inference, place `deeplabv3plus_resnet50.onnx` and the input video in the project root.

> 💡 `export_onnx.py` does **not** require `--gpus all`. `main.py` does: it intentionally stops if CUDA is unavailable rather than silently falling back to CPU.

### 1. Build the image 🏗️

Run from the `lane_segment` project folder.

**Linux / macOS shell**

```bash
docker build -t lane-segment .
```

**Windows PowerShell**

```powershell
docker build -t lane-segment .
```

### 2. Export the Keras model to ONNX 🔄

This reads `bestmodel.keras` and creates `deeplabv3plus_resnet50.onnx` in the same project folder.

**Linux / macOS shell**

```bash
docker run --rm -v "$PWD:/data" --entrypoint python3 lane-segment /app/export_onnx.py
```

**Windows PowerShell**

```powershell
docker run --rm -v "${PWD}:/data" --entrypoint python3 lane-segment /app/export_onnx.py
```

### 3. Run GPU video inference 🎥

Replace `input.mp4` and `output.mp4` with filenames in the project root. Arguments must be filenames, not paths.

**Linux / macOS shell**

```bash
docker run --rm --gpus all -v "$PWD:/data" lane-segment input.mp4 output.mp4
```

**Windows PowerShell**

```powershell
docker run --rm --gpus all -v "${PWD}:/data" lane-segment input.mp4 output.mp4
```

On successful startup, the logs show `CUDAExecutionProvider`; the green-overlay video is written to the mounted project folder. 🎉

## 📁 Project contents

| File | Purpose |
| --- | --- |
| `train.ipynb` | Dataset setup, augmentation, custom DeepLabV3+ training, and evaluation |
| `bestmodel.keras` | Best Keras checkpoint selected by validation Mean IoU — input to the ONNX exporter |
| `export_onnx.py` | Keras-to-ONNX conversion and ONNX validation |
| `deeplabv3plus_resnet50.onnx` | ONNX model used for inference |
| `main.py` | CUDA ONNX Runtime video segmentation runner |
| `Dockerfile` | Unified image for conversion and GPU inference |
| `1.png`, `2.png`, `3.png` | Example segmentation outputs |
| `video1.mp4`, `video2.mp4`, `video3.mp4` | Example videos showing lane segmentation inference |

---

⭐ Keep the model, input data, and output files together in the mounted project folder when running Docker.