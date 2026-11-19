import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


import json
import base64
import cv2
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.applications.resnet50 import preprocess_input
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# import gradcam functions
from gradcam import generate_gradcam, overlay_heatmap


# -------------------------
# Flask Setup
# -------------------------
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)


# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "resnet50_pneumonia.h5")

# Upload directories
UPLOAD_NORMAL = os.path.join(BASE_DIR, "uploads", "normal")
UPLOAD_PNEUMONIA = os.path.join(BASE_DIR, "uploads", "pneumonia")
os.makedirs(UPLOAD_NORMAL, exist_ok=True)
os.makedirs(UPLOAD_PNEUMONIA, exist_ok=True)


# -------------------------
# Load Model
# -------------------------
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully")


# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
PNEUMONIA_THRESHOLD = 0.6  # Only classify as PNEUMONIA above 60% confidence


# -------------------------
# Image Preprocessing
# -------------------------
def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32)

    # Use ResNet50's preprocess_input to match training preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


# -------------------------
# Prediction API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:

        # read image
        image = Image.open(file).convert("RGB")

        processed = preprocess_image(image)

        # -------------------------
        # Model Prediction
        # -------------------------
        prediction = model.predict(processed)

        prob = float(prediction[0][0])

        label = "PNEUMONIA" if prob > PNEUMONIA_THRESHOLD else "NORMAL"

        # Confidence: how sure the model is about its chosen label
        # If PNEUMONIA: confidence = prob
        # If NORMAL: confidence = 1 - prob (how sure it's NOT pneumonia)
        confidence = prob if label == "PNEUMONIA" else (1.0 - prob)

        # -------------------------
        # GradCAM Heatmap
        # -------------------------
        heatmap = generate_gradcam(model, processed)

        original = cv2.cvtColor(
            np.array(image.resize((IMG_SIZE, IMG_SIZE))),
            cv2.COLOR_RGB2BGR
        )

        overlay = overlay_heatmap(original, heatmap)

        # -------------------------
        # Convert heatmap to base64
        # -------------------------
        _, buffer = cv2.imencode(".jpg", overlay)

        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

        # -------------------------
        # Save uploaded image to uploads folder
        # -------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        save_name = f"{timestamp}{ext}"

        if label == "PNEUMONIA":
            save_path = os.path.join(UPLOAD_PNEUMONIA, save_name)
        else:
            save_path = os.path.join(UPLOAD_NORMAL, save_name)

        image.save(save_path)

        # -------------------------
        # Response
        # -------------------------
        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "heatmap": heatmap_base64
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -------------------------
# Model Metrics API
# -------------------------
@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Serve F1 score, recall, precision, and other evaluation metrics."""
    metrics_path = os.path.join(BASE_DIR, "model", "metrics.json")

    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not available. Run evaluate_model.py first."}), 404

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Web Routes for Frontend
# -------------------------
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":

    print("Starting Pneumonia Detection API...")

    app.run(debug=True, use_reloader=False)