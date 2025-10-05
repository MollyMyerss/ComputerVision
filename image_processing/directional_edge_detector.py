import numpy as np
from typing import Tuple
from calculate_gradient import calculate_gradient

# Detect edges in a specific direction range (in degrees [0, 180])
def directional_edge_detector(img: np.ndarray, direction_range: Tuple[float, float]) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    low, high = direction_range
    if low > high:
        low, high = high, low
    mag, ang = calculate_gradient(img)
    ang180 = np.where(ang < 0, ang + 180.0, ang)
    mask_dir = (ang180 >= low) & (ang180 <= high) & (mag > 0)
    return (mask_dir.astype(np.uint8) * 255)
