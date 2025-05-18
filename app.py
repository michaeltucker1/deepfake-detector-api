from pathlib import Path
from io import BytesIO
from functools import lru_cache
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

BASE_DIR     = Path(__file__).parent
MODELS_ROOT  = BASE_DIR / "models"
TARGET_SIZE  = (256, 256)
THRESHOLD    = 0.65
ALLOWED_IDS  = {"v1", "v2"}          
DEFAULT_ID   = "v2"

app = Flask(__name__)
CORS(app)

def model_path(model_id: str) -> Path:
    """Return the SavedModel directory (or .h5) for the given ID."""
    saved_dir = MODELS_ROOT / f"saved_model_{model_id}"
    if saved_dir.is_dir():
        return saved_dir
    h5_path = MODELS_ROOT / f"saved_model_{model_id}.h5"
    if h5_path.is_file():
        return h5_path
    raise FileNotFoundError(f"Model files for '{model_id}' not found.")

@lru_cache(maxsize=4)
def load_model(model_id: str):
    """Load & memoise the TF model so we don’t hit disk every request."""
    return tf.keras.models.load_model(model_path(model_id))

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/api/predict", methods=["POST"])
def predict():
    model_id = request.args.get("model", DEFAULT_ID).lower()
    if model_id not in ALLOWED_IDS:
        return jsonify({"error": f"Unknown model '{model_id}'"}), 400
    
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "File is not a valid image"}), 400

    x = preprocess(img)

    try:
        model = load_model(model_id)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    prob = float(model.predict(x, verbose=0)[0][0])

    return jsonify(
        {
            "model": model_id,
            "is_deepfake": prob >= THRESHOLD,
            "confidence": round(prob, 4),
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
