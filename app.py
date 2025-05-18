from pathlib import Path
from io import BytesIO
from functools import lru_cache
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

BASE_DIR    = Path(__file__).parent
MODELS_ROOT = BASE_DIR / "models"          
ALLOWED_IDS = {"v1", "v2"}                 
DEFAULT_ID  = "v1"
THRESHOLD   = 0.5

app = Flask(__name__)
CORS(app)

def _model_path(model_id: str) -> Path:
    """Return SavedModel dir or .h5 file for a given model id."""
    saved_dir = MODELS_ROOT / f"saved_model_{model_id}"
    if saved_dir.is_dir():
        return saved_dir
    h5_file = MODELS_ROOT / f"saved_model_{model_id}.h5"
    if h5_file.is_file():
        return h5_file
    raise FileNotFoundError(f"No model files found for id '{model_id}'")

@lru_cache(maxsize=4)
def load_model_and_size(model_id: str):
    """
    Load model once and memoise.  
    Returns (model, (height, width)).
    """
    m = tf.keras.models.load_model(_model_path(model_id))

    # Keras models can have multiple inputs; handle list/tuple
    first_shape = m.input_shape[0] if isinstance(m.input_shape, (list, tuple)) else m.input_shape
    # first_shape: (None, H, W, C)  for channels-last
    h, w = int(first_shape[1]), int(first_shape[2])

    if not h or not w:
        raise ValueError(f"Could not determine input size for model '{model_id}' (shape={first_shape})")

    return m, (h, w)

def preprocess(img: Image.Image, target_wh: tuple[int, int]) -> np.ndarray:
    img = img.resize(target_wh)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/api/predict", methods=["POST"])
def predict():
    model_id = request.args.get("model", DEFAULT_ID).lower()
    if model_id not in ALLOWED_IDS:
        return jsonify({"error": f"Unknown model '{model_id}'"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(BytesIO(f.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "File is not a valid image"}), 400

    try:
        model, input_wh = load_model_and_size(model_id)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 500

    x = preprocess(img, input_wh)

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
