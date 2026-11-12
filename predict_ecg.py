import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# PATHS & SETTINGS
# ============================================================

# 1. Point directly to the .keras file saved in Cell 16 (# 17)
MODEL_PATH = r"G:\ECG-prototype\ecg_densenet121.keras"
CLASS_NAMES_PATH = r"G:\ECG-prototype\class_names.json"

# 2. Match the (384, 384) training resolution from Cell 2
IMAGE_SIZE = (384, 384)

# ============================================================
# LOAD KERAS MODEL & CLASS NAMES
# ============================================================

print("Loading ECG model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image).astype("float32")

    # Apply contrast adjustment and clip in [0, 255] first, then rescale
    # so the pipeline matches ImageDataGenerator's exact order of operations
    image = tf.image.adjust_contrast(image, contrast_factor=1.3)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = image / 255.0

    image = tf.expand_dims(image, axis=0)
    return image

# ============================================================
# PREDICTION
# ============================================================

def predict_ecg(image_path):
    image = preprocess_image(image_path)

    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index] * 100

    print("\n" + "=" * 75)
    print("                           ECG RESULT")
    print("=" * 75)
    print(f"Predicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.2f}%\n")
    print("Class probabilities:")
    print("-" * 75)

    for name, probability in zip(class_names, probabilities):
        print(f"{name:<62} {probability * 100:>7.2f}%")

    print("=" * 75)

# ============================================================
# MAIN
# ============================================================

if len(sys.argv) != 2:
    print('\nUsage:\npython predict_ecg.py "path_to_ecg_image"')
    print('\nExample:\npython predict_ecg.py "D:\\ecg\\example.jpg"')
else:
    image_path = sys.argv[1]
    try:
        predict_ecg(image_path)
    except FileNotFoundError:
        print(f"\nERROR: File not found:\n{image_path}")
    except Exception as e:
        print(f"\nERROR while predicting:\n{e}")