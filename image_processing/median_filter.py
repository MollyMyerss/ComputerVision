import numpy as np

def median_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a median filter of window `size` x `size` to a grayscale image."""
    if img is None:
        raise ValueError("img is None")
    if size % 2 == 0 or size < 1:
        raise ValueError("size must be a positive odd integer")
    img_u8 = img.astype(np.uint8, copy=False)
    pad = size // 2
    padded = np.pad(img_u8, pad_width=pad, mode='reflect')
    H, W = img_u8.shape
    out = np.empty_like(img_u8)
    for i in range(H):
        for j in range(W):
            window = padded[i:i+size, j:j+size]
            out[i, j] = np.median(window)
    return out
