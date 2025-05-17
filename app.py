from pathlib import Path
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_DIR = Path(__file__).parent / "model" / "saved_model"  # or e.g. model.h5
TARGET_SIZE = (256, 256)  
THRESHOLD = 0.5           

app = Flask(__name__)
CORS(app)

try:
    if MODEL_DIR.is_dir():
        model = tf.keras.models.load_model(MODEL_DIR)
    else:
        model = tf.keras.models.load_model(str(MODEL_DIR)+".h5")
except Exception as exc:
    raise RuntimeError(f"Could not load TF model: {exc} Expected at {MODEL_DIR}")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0  # same rescale as ImageDataGenerator
    return np.expand_dims(arr, axis=0)

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "File is not a valid image"}), 400

    x = preprocess(img)
    prob = float(model.predict(x, verbose=0)[0][0])

    return jsonify({
        "is_deepfake": prob >= THRESHOLD,
        "confidence": round(prob, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)