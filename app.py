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
THRESHOLD   = 0.6

app = Flask(__name__)
CORS(app)

MODEL_SIZE_OVERRIDES = {       
    "v1": (256, 256),
    "v2": (224, 224),
}

def _model_path(model_id: str) -> Path:
    saved_dir = MODELS_ROOT / f"saved_model_{model_id}"
    if saved_dir.is_dir():
        return saved_dir
    h5_file = MODELS_ROOT / f"saved_model_{model_id}.h5"
    if h5_file.is_file():
        return h5_file
    raise FileNotFoundError(f"No model files found for id '{model_id}'")

def _discover_hw(m: tf.keras.Model) -> tuple[int | None, int | None]:
    """
    Try to find a concrete (H,W) inside the model graph.
    Returns (None, None) if nothing conclusive is found.
    """
    # 1) Preferred: model.inputs
    if m.inputs:
        shp = m.inputs[0].shape
        if shp[1] and shp[2]:
            return int(shp[1]), int(shp[2])

    # 2) Walk layers
    for layer in m.layers:
        if hasattr(layer, "input_shape") and layer.input_shape is not None:
            shp = layer.input_shape
            if isinstance(shp, (list, tuple)):
                shp = shp[0]            # handle nested lists
            if shp is not None and len(shp) >= 3 and shp[1] and shp[2]:
                return int(shp[1]), int(shp[2])
    return None, None  # no luck

@lru_cache(maxsize=4)
def load_model_and_size(model_id: str):
    """Load model once and memoise. Returns (model, (H,W))."""
    m = tf.keras.models.load_model(_model_path(model_id))

    h, w = _discover_hw(m)

    # Fall back to overrides
    if (h is None or w is None) and model_id in MODEL_SIZE_OVERRIDES:
        h, w = MODEL_SIZE_OVERRIDES[model_id]

    if h is None or w is None:
        raise ValueError(
            f"Could not determine input size for model '{model_id}'. "
            "Add it to MODEL_SIZE_OVERRIDES."
        )

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
