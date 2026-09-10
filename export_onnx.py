# ============================================
# Keras (.keras) → ONNX
# DeepLabV3+ + ResNet50
# ============================================


import os
import tensorflow as tf
import tf2onnx
import onnx


# ============================================
# Paths
# ============================================

# The default /data directory is mounted from the project folder by Docker.
# Environment variables allow other locations when running outside Docker.
KERAS_MODEL_PATH = os.environ.get("KERAS_MODEL_PATH", "/data/bestmodel.keras")
ONNX_MODEL_PATH = os.environ.get(
    "ONNX_MODEL_PATH", "/data/deeplabv3plus_resnet50.onnx"
)


print("=" * 70)
print("Keras → ONNX Conversion")
print("=" * 70)

print(f"Keras model:")
print(f"  {KERAS_MODEL_PATH}")

print(f"\nONNX output:")
print(f"  {ONNX_MODEL_PATH}")


# ============================================
# 1. Load Keras Model
# ============================================

print("\n[1] Loading Keras model...")

model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    compile=False
)

print("✅ Keras model loaded successfully")

print(f"   Input shape : {model.input_shape}")
print(f"   Output shape: {model.output_shape}")


# ============================================
# 2. Input Signature
# ============================================

print("\n[2] Creating input signature...")

input_signature = [
    tf.TensorSpec(
        shape=[None, 640, 640, 3],
        dtype=tf.float32,
        name="input"
    )
]

print("✅ Input signature created")


# ============================================
# 3. Convert Keras → ONNX
# ============================================

print("\n[3] Converting Keras → ONNX...")
print("⏳ Please wait...")

model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=17,
    output_path=ONNX_MODEL_PATH
)

print("✅ Conversion completed successfully!")


# ============================================
# 4. Validate ONNX
# ============================================

print("\n[4] Validating ONNX model...")

onnx_model = onnx.load(ONNX_MODEL_PATH)

onnx.checker.check_model(onnx_model)

print("✅ ONNX model is valid!")


# ============================================
# 5. Model Information
# ============================================

file_size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 ** 2)

print("\n" + "=" * 70)
print("🎉 ONNX CONVERSION COMPLETE")
print("=" * 70)

print(f"\n📁 ONNX file:")
print(f"   {ONNX_MODEL_PATH}")

print(f"\n📦 File size:")
print(f"   {file_size_mb:.2f} MB")

print("\n🔹 Inputs:")

for inp in onnx_model.graph.input:
    print(f"   Name : {inp.name}")

print("\n🔹 Outputs:")

for out in onnx_model.graph.output:
    print(f"   Name : {out.name}")

print("\n🔹 Expected input:")
print("   [Batch, 640, 640, 3]")

print("\n🔹 Expected output:")
print("   [Batch, 640, 640, 2]")

print("\n✅ Conversion finished successfully!")
print("=" * 70)
