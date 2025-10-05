import numpy as np

# Equalize histogram of a grayscale image using cumulative distribution function
def equalize_histogram(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    img_u8 = img.astype(np.uint8, copy=False)
    hist, _ = np.histogram(img_u8, bins=256, range=(0, 256))
    cdf = hist.cumsum()

    cdf_masked = np.ma.masked_equal(cdf, 0)
    if cdf_masked.mask.all():
        return np.zeros_like(img_u8)

    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    cdf_scaled = (cdf_masked - cdf_min) * 255.0 / (cdf_max - cdf_min)
    lut = np.ma.filled(cdf_scaled, 0).astype(np.uint8)
    return lut[img_u8]
