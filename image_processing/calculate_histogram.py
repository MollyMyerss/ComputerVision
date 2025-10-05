import numpy as np
from typing import Tuple

# Calculate histogram and normalized distribution of pixel intensities
def calculate_histogram(img: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    if img is None:
        raise ValueError("img is None")
    if bins <= 0:
        raise ValueError("bins must be positive")
    # Ensure grayscale
    img = img.astype(np.uint8, copy=False)
    counts, _ = np.histogram(img, bins=bins, range=(0, 256))
    total = counts.sum()
    dist = counts.astype(np.float64) / total if total > 0 else np.zeros_like(counts, dtype=np.float64)
    return counts, dist
