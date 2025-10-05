import numpy as np

# Contrast stretch to full [0,255] or given [r_min,r_max] range
def contrast_stretch(img: np.ndarray, r_min=None, r_max=None) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    img = img.astype(np.float32, copy=False)

    if r_min is None:
        r_min = float(np.min(img))
    if r_max is None:
        r_max = float(np.max(img))
    if r_max <= r_min:
        return np.zeros_like(img, dtype=np.uint8)

    clipped = np.clip(img, r_min, r_max)
    scaled = (clipped - r_min) * (255.0 / (r_max - r_min))
    return np.clip(scaled, 0, 255).astype(np.uint8)
