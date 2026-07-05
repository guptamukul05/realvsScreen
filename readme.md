# 🕵️ Spot the Fake Photo

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Live%20Demo-black?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A lightweight computer vision system that detects whether a photo is of a **real, physical object** or a **recapture** (a photo taken of a screen or a printed photo). Built without deep learning, using classic signal-processing and image forensics techniques instead.

## 🧩 The Problem

Apps that rely on users submitting genuine photos (verification flows, claims processing, marketplace listings, etc.) are vulnerable to a simple form of cheating: instead of photographing the real thing, someone photographs a *picture* of the real thing on another screen or a printout. This project explores how far you can get at detecting that, using signal-level cues rather than object recognition.

## 🔬 Approach

Rather than training a convolutional neural network, this project treats the problem as **image forensics**. Every feature is designed around a specific physical reason a recaptured image differs from a genuine one:

- **Moiré / aliasing detection** — a screen's pixel grid interacts with a camera's sensor grid, producing frequency-domain artifacts that don't appear in natural photos. Detected via FFT analysis of the spatial frequency spectrum.
- **Glare and specular reflection detection** — glass and screen surfaces reflect ambient light in characteristic near-white, low-saturation blobs, identified using HSV color space analysis and connected-component detection.
- **Sharpness uniformity** — real 3D scenes have natural depth-of-field variation across the frame; a flat screen photographed head-on tends to be evenly focused. Measured via blockwise Laplacian variance.
- **Chromatic edge fringing** — a screen's subpixel structure causes slight misalignment between red, green, and blue edge maps at high-contrast boundaries, measured via per-channel edge comparison.

These features are combined using a small, regularized logistic regression, validated with k-fold cross-validation to keep the model honest and resistant to overfitting on a small dataset.

An optional module also supports adding a frozen, pretrained CNN embedding as an additional feature (transfer learning without fine-tuning), for anyone who wants to experiment with extending the feature set further.

## ✨ Features

- 🔍 **Forensic feature extraction** — pure numpy/opencv, no GPU required
- 📊 **Cross-validated training pipeline** with per-feature diagnostics to catch weak or redundant signals before they make it into the final model
- ⚡ **Fast, single-purpose predictor** — `predict.py image.jpg` returns one score, ready to integrate anywhere
- 🎥 **Live camera demo** — a small local web app that scores frames straight from your webcam in real time
- 🧩 **Modular design** — features, training, and inference are cleanly separated, making it easy to add, test, or remove individual signals

## 📁 Project Structure

```
spotfake/
├── features.py       # Core forensic feature extraction
├── embedding.py       # Optional frozen pretrained-CNN feature (transfer learning)
├── train.py          # Fits the model with cross-validation
├── predict.py         # Command-line predictor: predict.py image.jpg
├── diagnose.py         # Per-feature diagnostics (separation stats, outliers)
├── app.py             # Flask backend for the live camera demo
├── templates/
│   └── index.html      # Live demo front end
├── requirements.txt
├── real/              # Your real-object training photos go here
└── screen/            # Your screen/recapture training photos go here
```

## 🚀 Getting Started

### 0. Clone and set up a virtual environment (recommended)

```bash
git clone <your-repo-url>
cd project5
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your training photos

Put real-object photos in `real/` and screen/recapture photos in `screen/`. More variety in lighting, angle, and device type produces a more robust model.

### 3. Train

```bash
python train.py --real real/ --screen screen/
```

This prints cross-validated accuracy, a confusion matrix, and saves the fitted model to `weights.json`.

### 4. Predict

```bash
python predict.py path/to/image.jpg
```

Outputs a single score from 0 to 1: closer to 0 means real, closer to 1 means a screen recapture.

### 5. (Optional) Diagnose feature quality

```bash
python diagnose.py --real real/ --screen screen/
```

Shows how well each individual feature separates the two classes on your data, useful before adding or removing features.

### 6. (Optional) Run the live demo

```bash
pip install flask
python app.py
```

Then open `http://localhost:5000` in a browser, allow camera access, and click "Start Live Scoring" to see predictions update live from your webcam.

## 🛠️ Extending This Project

- Add new features to `features.py` and check their impact with `diagnose.py` before wiring them into the model — this keeps the pipeline honest and avoids adding noise disguised as signal.
- Enable the optional pretrained-embedding feature in `train.py` with the `--use-embedding` flag if you want to experiment with transfer learning as an additional signal source.
- The modular structure makes it straightforward to swap in a different classifier, add new forensic cues, or adapt the pipeline to related image-authenticity problems.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.