import numpy as np
from typing import Optional
from calculate_gradient import calculate_gradient

def _normalize_to_uint8(mag: np.ndarray) -> np.ndarray:
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax <= 1e-6:
        return np.zeros_like(mag, dtype=np.uint8)
    scaled = (mag / mmax) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

# Perform Sobel edge detection with thresholding
def sobel_edge_detector(img: np.ndarray, threshold: int) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be in [0, 255]")
    mag, _ = calculate_gradient(img)
    mag_u8 = _normalize_to_uint8(mag)
    edges = (mag_u8 >= threshold).astype(np.uint8) * 255
    return edges
