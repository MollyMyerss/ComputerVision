import numpy as np
from typing import Tuple

def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    assert kh == kw and kh % 2 == 1, "Kernel must be odd-sized and square"
    pad = kh // 2

    x = np.pad(image.astype(np.float32, copy=False),
               ((pad, pad), (pad, pad)), mode="edge")

    H, W = image.shape
    s0, s1 = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(H, W, kh, kw), strides=(s0, s1, s0, s1), writeable=False
    )
    k = kernel[::-1, ::-1].astype(np.float32, copy=False)  
    out = np.tensordot(windows, k, axes=([2, 3], [0, 1]))  
    return out 

# --- Sobel kernels ---
_SX = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float32)

_SY = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]], dtype=np.float32)

# Compute gradient magnitude and angle using Sobel operators
def calculate_gradient(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if img is None:
        raise ValueError("img is None")
    gray = img.astype(np.float32, copy=False)

    gx = apply_convolution(gray, _SX)
    gy = apply_convolution(gray, _SY)

    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.degrees(np.arctan2(gy, gx))
    return mag, ang
