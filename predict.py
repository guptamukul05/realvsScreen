#!/usr/bin/env python3


import json
import os
import sys
import time

import numpy as np
import cv2

from features import extract_features, FEATURE_NAMES

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


def _load_weights():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"{WEIGHTS_PATH} not found. Run `python train.py --real real/ --screen screen/` "
            "first to fit weights on your labeled photos."
        )
    with open(WEIGHTS_PATH) as f:
        return json.load(f)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(path_or_array, weights=None):
    if weights is None:
        weights = _load_weights()

    if isinstance(path_or_array, str):
        bgr = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path_or_array}")
    else:
        bgr = path_or_array  # already a loaded BGR numpy array (e.g. from a webcam frame)

    feats = extract_features(bgr)
    values = [feats[k] for k in FEATURE_NAMES]

    if weights.get("use_embedding"):
        import embedding as emb_mod
        e = emb_mod.get_embedding(bgr)
        ds = emb_mod.deep_score(e, weights["real_centroid"], weights["screen_centroid"])
        values.append(ds)

    x = np.array(values, dtype=np.float64)
    mean = np.array(weights["mean"])
    std = np.array(weights["std"])
    xs = (x - mean) / std

    coef = np.array(weights["coef"])
    z = float(np.dot(coef, xs) + weights["intercept"])
    score = float(_sigmoid(z))
    return score, feats


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py some_image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    weights = _load_weights()

    t0 = time.time()
    score, feats = predict(img_path, weights)
    t1 = time.time()

    print(f"{score:.4f}")
    sys.stderr.write(f"[info] latency: {(t1 - t0)*1000:.1f} ms | features: {feats}\n")


if __name__ == "__main__":
    main()