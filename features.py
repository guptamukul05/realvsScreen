
import cv2
import numpy as np

# Disable OpenCV's internal multi-threading. In some environments/builds,
# Canny + connected-components under threading can segfault on
# pathological (highly repetitive/periodic) input. Single-threaded is
# also more predictable for latency benchmarking, which we care about.
cv2.setNumThreads(1)

FEATURE_NAMES = [
    "moire_score",
    "glare_score",
    "sharpness_uniformity",
    "fringe_score",
]

MAX_DIM = 900  # resize cap for speed; forensic signals survive downscaling fine


def _load_and_resize(path_or_array):
    if isinstance(path_or_array, (str, bytes)):
        img = cv2.imread(str(path_or_array), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path_or_array}")
    else:
        img = path_or_array

    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _moire_score(gray):
    
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_int = r.astype(np.int32)

    # radial average = the expected "natural falloff" magnitude at each radius
    counts = np.bincount(r_int.ravel())
    sums = np.bincount(r_int.ravel(), weights=mag.ravel())
    radial_mean = sums / np.maximum(counts, 1)
    expected = radial_mean[r_int]

    # residual = how much a pixel exceeds what's normal for its radius
    residual = mag - expected

    r_norm = r / r.max()
    band_mask = (r_norm > 0.12) & (r_norm < 0.45)
    band_residual = residual[band_mask]

    if band_residual.size == 0:
        return 0.0

    # peakiness: strong, concentrated positive spikes vs. general spread
    peak = np.percentile(band_residual, 99)
    spread = band_residual.std() + 1e-6
    return float(peak / spread)


def _glare_score(bgr):
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s < 60) & (v > 230)).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_labels <= 1:
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    total_px = mask.size
    largest_frac = areas.max() / total_px if areas.size else 0.0
    coverage_frac = areas.sum() / total_px

    return float(3.0 * largest_frac + 1.5 * coverage_frac)


def _sharpness_uniformity(gray):
    
    h, w = gray.shape
    n = 6
    bh, bw = h // n, w // n
    if bh < 8 or bw < 8:
        return 0.0

    vals = []
    for i in range(n):
        for j in range(n):
            block = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            lap = cv2.Laplacian(block, cv2.CV_64F)
            vals.append(lap.var())

    vals = np.array(vals)
    mean_v = vals.mean() + 1e-6
    cv_ = vals.std() / mean_v  # coefficient of variation

    # LOW coefficient of variation -> uniform sharpness -> screen-like
    # convert to a "screen-likeness" score by inverting
    uniformity = 1.0 / (1.0 + cv_)
    return float(uniformity)


def _fringe_score(bgr):
    
    b, g, r = cv2.split(bgr)

    def edge_mask(ch):
        return cv2.Canny(ch, 60, 160) > 0

    eb, eg, er = edge_mask(b), edge_mask(g), edge_mask(r)

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / union if union > 0 else 1.0

    ious = [iou(er, eg), iou(eg, eb), iou(er, eb)]
    return float(1.0 - np.mean(ious))


def _edge_orientation_entropy(gray):
    
    edges = cv2.Canny(gray, 60, 160)
    if edges.sum() == 0:
        return 0.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mask = edges > 0
    angles = np.arctan2(gy[mask], gx[mask])

    if angles.size < 10:
        return 0.0

    hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
    p = hist.astype(np.float64) / (hist.sum() + 1e-8)
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    max_entropy = np.log(18)
    normalized = entropy / max_entropy

    return float(1.0 - normalized)


def _edge_periodicity_score(gray):
    
    edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
    edges = edges - edges.mean()

    # autocorrelation via Wiener-Khinchin theorem (fast, avoids O(n^2) loops)
    f = np.fft.fft2(edges)
    power = f * np.conj(f)
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)

    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    center_val = autocorr[cy, cx] + 1e-8
    ac_norm = autocorr / center_val  # normalize so zero-lag peak = 1.0

    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    # small-to-medium lag ring, excluding the (trivial) zero-lag peak itself
    ring_mask = (r > 3) & (r < 40)
    ring_vals = np.abs(ac_norm[ring_mask])

    if ring_vals.size == 0:
        return 0.0

    # peakiness: a real repeating structure produces a strong secondary
    # peak in this ring; natural edge patterns decay smoothly instead
    return float(np.percentile(ring_vals, 99))


def extract_features(path_or_array):
    """Returns a dict of the 5 named features for one image."""
    bgr = _load_and_resize(path_or_array)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    return {
        "moire_score": _moire_score(gray),
        "glare_score": _glare_score(bgr),
        "sharpness_uniformity": _sharpness_uniformity(gray),
        "fringe_score": _fringe_score(bgr),
    }


def feature_vector(path_or_array):
    d = extract_features(path_or_array)
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float64)