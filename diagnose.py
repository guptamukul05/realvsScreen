import argparse
import glob
import os
import numpy as np
from features import extract_features, FEATURE_NAMES
from pathlib import Path
def load_folder(folder):
    # Read each file only once (case-insensitive)
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rows = []
    for p in paths:
        try:
            feats = extract_features(str(p))
            rows.append((str(p), feats))
        except Exception as e:
            print(f"  [skip] {p}: {e}")
    return rows
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real")
    ap.add_argument("--screen", default="screen")
    args = ap.parse_args()
    real_rows = load_folder(args.real)
    screen_rows = load_folder(args.screen)
    print(f"\nLoaded {len(real_rows)} real, {len(screen_rows)} screen\n")
    print("\nChecking for duplicate files...\n")
    real_names = [str(p) for p, _ in real_rows]
    screen_names = [str(p) for p, _ in screen_rows]
    print("Unique real files:", len(set(real_names)))
    print("Total real files :", len(real_names))
    print("Unique screen files:", len(set(screen_names)))
    print("Total screen files :", len(screen_names))
    print(f"{'feature':25s} {'real mean±std':>20s} {'screen mean±std':>20s} {'separation':>12s}")
    print("-" * 82)
    for name in FEATURE_NAMES:
        real_vals = np.array([f[name] for _, f in real_rows])
        screen_vals = np.array([f[name] for _, f in screen_rows])
        rm, rs = real_vals.mean(), real_vals.std()
        sm, ss = screen_vals.mean(), screen_vals.std()
        pooled_std = np.sqrt((rs**2 + ss**2) / 2) + 1e-8
        cohens_d = (sm - rm) / pooled_std  # >0 means screen > real, as designed
        print(f"{name:25s} {rm:8.3f}±{rs:<10.3f} {sm:8.3f}±{ss:<10.3f} {cohens_d:>+11.2f}")
    print("\n'separation' is Cohen's d: positive means screen photos score higher")
    print("(as each feature is designed to expect). Near 0 = feature isn't")
    print("distinguishing your two folders. Negative = feature is backwards on")
    print("your data specifically -- worth looking at WHY (see the 5 highest/")
    print("lowest scoring images per feature to eyeball what's happening).\n")
    # Show extremes per feature to help you eyeball actual images
    for name in FEATURE_NAMES:
        all_rows = [(p, f[name], "real") for p, f in real_rows] + \
                   [(p, f[name], "screen") for p, f in screen_rows]
        all_rows.sort(key=lambda x: x[1])
        print(f"\n-- {name} --")
        print("  lowest 3:", [(os.path.basename(p), round(v, 3), lbl) for p, v, lbl in all_rows[:3]])
        print("  highest 3:", [(os.path.basename(p), round(v, 3), lbl) for p, v, lbl in all_rows[-3:]])
if __name__ == "__main__":
    main()