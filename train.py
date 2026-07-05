import argparse
import glob
import json
import os
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from features import extract_features, FEATURE_NAMES
from pathlib import Path
def load_folder(folder, label):
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    X, y, used_paths, raw_imgs = [], [], [], []
    for p in paths:
        try:
            import cv2
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("could not read")
            feats = extract_features(str(p))
            X.append([feats[k] for k in FEATURE_NAMES])
            y.append(label)
            used_paths.append(str(p))
            raw_imgs.append(bgr)
        except Exception as e:
            print(f"  [skip] {p}: {e}")
    return X, y, used_paths, raw_imgs
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real")
    ap.add_argument("--screen", default="screen")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="weights.json")
    ap.add_argument("--use-embedding", action="store_true",
                     help="Add frozen pretrained CNN similarity feature (requires torch)")
    args = ap.parse_args()
    print(f"Loading REAL photos from {args.real}/ ...")
    X_real, y_real, p_real, img_real = load_folder(args.real, 0)
    print(f"  {len(X_real)} real photos loaded")
    print(f"Loading SCREEN photos from {args.screen}/ ...")
    X_screen, y_screen, p_screen, img_screen = load_folder(args.screen, 1)
    print(f"  {len(X_screen)} screen photos loaded")
    X = np.array(X_real + X_screen, dtype=np.float64)
    y = np.array(y_real + y_screen, dtype=np.int64)
    paths = p_real + p_screen
    raw_imgs = img_real + img_screen
    if len(X) < 20:
        print("WARNING: fewer than 20 total images. Accuracy numbers will be noisy.")
    embeddings = None
    if args.use_embedding:
        print("\nComputing frozen CNN embeddings (first run downloads pretrained weights)...")
        import embedding as emb_mod
        embeddings = np.array([emb_mod.get_embedding(im) for im in raw_imgs])
        print(f"  embeddings shape: {embeddings.shape}")
    n_folds = min(args.folds, np.bincount(y).min())
    if n_folds < 2:
        raise ValueError("Need at least 2 examples of each class per fold. Add more data.")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        if args.use_embedding:
            import embedding as emb_mod
            emb_train = embeddings[train_idx]
            real_centroid = emb_train[y_train == 0].mean(axis=0)
            screen_centroid = emb_train[y_train == 1].mean(axis=0)
            deep_train = np.array([
                emb_mod.deep_score(e, real_centroid, screen_centroid) for e in emb_train
            ]).reshape(-1, 1)
            deep_val = np.array([
                emb_mod.deep_score(e, real_centroid, screen_centroid) for e in embeddings[val_idx]
            ]).reshape(-1, 1)
            X_train = np.hstack([X_train, deep_train])
            X_val = np.hstack([X_val, deep_val])
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_s = (X_train - mean) / std
        X_val_s = (X_val - mean) / std
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        clf.fit(X_train_s, y_train)
        oof_proba[val_idx] = clf.predict_proba(X_val_s)[:, 1]
    y_pred = (oof_proba >= 0.5).astype(int)
    cv_acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"\n== {n_folds}-fold cross-validated accuracy: {cv_acc*100:.1f}% ==")
    print("Confusion matrix [rows=true, cols=pred] (0=real, 1=screen):")
    print(cm)
    wrong = np.where(y_pred != y)[0]
    if len(wrong):
        print("\nMisclassified images:")
        for i in wrong:
            print(f"  {paths[i]}  true={y[i]}  pred_prob_screen={oof_proba[i]:.2f}")
    # ---- Fit FINAL model on all data for deployment ----
    feature_names_final = list(FEATURE_NAMES)
    X_final = X.copy()
    real_centroid_final = screen_centroid_final = None
    if args.use_embedding:
        import embedding as emb_mod
        real_centroid_final = embeddings[y == 0].mean(axis=0)
        screen_centroid_final = embeddings[y == 1].mean(axis=0)
        deep_final = np.array([
            emb_mod.deep_score(e, real_centroid_final, screen_centroid_final) for e in embeddings
        ]).reshape(-1, 1)
        X_final = np.hstack([X_final, deep_final])
        feature_names_final = feature_names_final + ["deep_score"]
    mean = X_final.mean(axis=0)
    std = X_final.std(axis=0) + 1e-8
    X_final_s = (X_final - mean) / std
    clf_final = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf_final.fit(X_final_s, y)
    t0 = time.time()
    _ = extract_features(paths[0])
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000
    if args.use_embedding:
        import embedding as emb_mod
        t0 = time.time()
        _ = emb_mod.get_embedding(raw_imgs[0])
        t1 = time.time()
        latency_ms += (t1 - t0) * 1000
    weights = {
        "feature_names": feature_names_final,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": clf_final.coef_[0].tolist(),
        "intercept": float(clf_final.intercept_[0]),
        "threshold": 0.5,
        "cv_accuracy": cv_acc,
        "n_train_images": int(len(X)),
        "n_folds": n_folds,
        "measured_feature_latency_ms": latency_ms,
        "use_embedding": bool(args.use_embedding),
        "real_centroid": real_centroid_final.tolist() if real_centroid_final is not None else None,
        "screen_centroid": screen_centroid_final.tolist() if screen_centroid_final is not None else None,
    }
    with open(args.out, "w") as f:
        json.dump(weights, f)
    print(f"\nSaved weights to {args.out}")
    print(f"Latency (1 image, this machine): ~{latency_ms:.1f} ms")
    print("\nCoefficients:")
    for name, c in zip(feature_names_final, clf_final.coef_[0]):
        print(f"  {name:25s} {c:+.3f}")
if __name__ == "__main__":
    main()