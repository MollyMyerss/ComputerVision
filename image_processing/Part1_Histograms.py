import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from contrast_stretch import contrast_stretch
from equalize_histogram import equalize_histogram
from calculate_histogram import calculate_histogram

class Part1HistogramDemo:
    def __init__(self, img_path="images/LowContrast.jpg"):
        self.img_path = img_path

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:  
            img = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img

    def _hist256(self, a: np.ndarray) -> np.ndarray:
        counts, _ = calculate_histogram(a, 256)
        return counts

    def run(self, show: bool = True, save_dir: str | None = "outputs"):
        #read image
        img = iio.imread(self.img_path)
        img = self._to_gray(img)

        stretched = contrast_stretch(img, None, None)
        equalized = equalize_histogram(img)

        #histograms
        h_orig = self._hist256(img)
        h_st   = self._hist256(stretched)
        h_eq   = self._hist256(equalized)

        #plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].bar(np.arange(256), h_orig); axes[0].set_title("Original")
        axes[1].bar(np.arange(256), h_st);   axes[1].set_title("Stretched")
        axes[2].bar(np.arange(256), h_eq);   axes[2].set_title("Equalized")
        for ax in axes:
            ax.set_xlabel("Intensity"); ax.set_ylabel("Count")
        fig.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "p1_hist_summary.png")
            fig.savefig(out_path, dpi=200)
            print(f"[saved] ./{out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    Part1HistogramDemo(
        img_path="images/LowContrast.jpg"
    ).run(show=True, save_dir="outputs")
