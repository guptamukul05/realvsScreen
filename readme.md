# Spot the Fake Photo — Notes

## Approach

I didn't train a neural net for this. With only ~100 photos, a CNN (even a
fine-tuned MobileNet) is very likely to memorize backgrounds, lighting, or
which device I used, rather than the actual real-vs-screen signal — and it
still has to run on a phone. So I went with signal forensics instead:
5 handcrafted features, each tied to a physical reason screens look
different from real objects, fed into a small L2-regularized logistic
regression (5 weights + 1 bias, ~6 parameters total).

The 5 features:
- **Moiré/aliasing energy** — a screen's pixel grid beats against the
  camera sensor's grid, showing up as unnatural peaks in the mid-band of
  the 2D FFT. Real textures spread energy smoothly instead.
- **Highlight clipping** — backlit panels tend to blow out in hard,
  localized white patches; ambient light on real objects falls off more
  gradually.
- **Glare blobs** — glass reflects room light as near-white,
  low-saturation blobs (I measure their size/coverage).
- **Bezel line regularity** — long, straight, axis-aligned lines
  concentrated near the image border (Hough transform), which shows up
  when you photograph a screen or printout edge-on.
- **Sharpness uniformity** — a flat screen photographed head-on has
  unusually *even* focus across the frame; real 3D scenes have natural
  depth-of-field variation. I measure this as the coefficient of
  variation of blockwise Laplacian sharpness.

I fit the logistic regression on my own [N] photos ([N_real] real /
[N_screen] screen), used [K]-fold stratified cross-validation (so no
photo is ever scored by a model that saw it during training), and kept
the model deliberately tiny to avoid overfitting on such a small dataset.

## Accuracy

**Cross-validated accuracy: [XX.X]%** on my own [N]-photo set (this is
the honest, out-of-fold number from `train.py`, not a number computed
on training data). [N_wrong] images were misclassified — mostly [describe:
e.g. "a laptop screen photographed at a steep angle, which killed the
bezel-line signal" or "a real photo of a very flat, evenly-lit painting,
which fooled the sharpness-uniformity feature"].

I expect this to be somewhat lower on your held-out set since my data
covers a limited range of screens/lighting — see "what I'd improve"
below.

## Latency & cost

- **Latency:** ~[XX] ms per image on [device, e.g. "M-series MacBook
  CPU, single core"]. All 5 features are pure numpy/opencv ops (FFT,
  Canny, Hough, connected components) — no GPU, no network call.
- **Cost per image:** **$0 on-device** — this runs comfortably inside a
  phone's normal image-processing budget (no model weights to download
  beyond a ~1KB JSON file). If run server-side instead, at ~[XX]ms of
  CPU time per image, on a $0.02/vCPU-hour rough cloud rate that's
  roughly $[calc]/1,000 images or $[calc]/million — effectively
  negligible either way, since there's no GPU involved.

## What I'd improve with more time

- **More data diversity**: more screen types (OLED phone, LCD laptop,
  matte e-ink, printed photo, different resolutions) and more real-world
  textures that could fool the sharpness/line features (flat art,
  mirrors, glossy magazines).
- **Adaptive threshold**: right now I use 0.5; in production I'd tune the
  cutoff based on the cost of a false accept vs. false reject (see below).
- **Adversarial robustness**: as cheaters adapt (e.g., defocusing the
  camera slightly to kill moiré, or photographing at an angle to avoid
  the bezel-line feature), I'd rotate in new features rather than
  retraining the same 5 — e.g. chromatic subpixel fringing at edges, or
  comparing EXIF/sensor noise fingerprints where available.

## For "more experienced" question

- **Keeping it accurate as cheaters adapt**: treat this as a slowly
  escalating cat-and-mouse problem. Log borderline scores (say 0.3–0.7)
  for periodic human review, retrain on new confirmed cheats every few
  weeks, and keep 1-2 features held in reserve that aren't documented
  anywhere public, so a motivated cheater can't simply defeat all
  detectable signals at once.
- **Making it tiny/fast enough for a phone**: it already is — this is
  ~15-50ms of pure numpy/opencv per image with no model file beyond a
  tiny JSON of 6 numbers. The main phone-side cost would be decoding/
  resizing the image, which the camera pipeline is already doing.
- **Choosing the cutoff**: pick it based on the asymmetry of costs —
  falsely flagging a genuine user is more annoying than missing a rare
  cheat, so I'd bias the threshold toward fewer false positives (e.g.
  0.65-0.7 instead of 0.5) and use the 0.3–0.7 band as a "send to manual
  review" zone rather than a hard auto-decision either way.
