

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import predict as predict_mod

app = Flask(__name__)

# Load weights once at startup, not on every request (keeps latency honest)
_weights = predict_mod._load_weights()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    data_url = data.get("image", "")

    # strip the "data:image/jpeg;base64," prefix from the canvas capture
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if bgr is None:
        return jsonify({"error": "could not decode image"}), 400

    import time
    t0 = time.time()
    score, feats = predict_mod.predict(bgr, _weights)
    latency_ms = (time.time() - t0) * 1000

    label = "SCREEN (recapture)" if score >= 0.5 else "REAL photo"

    return jsonify({
        "score": round(score, 4),
        "label": label,
        "latency_ms": round(latency_ms, 1),
        "features": {k: round(v, 4) for k, v in feats.items()},
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)