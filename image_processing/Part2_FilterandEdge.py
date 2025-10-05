# part2_filterandedge.py
import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from median_filter import median_filter
from calculate_gradient import calculate_gradient

class Part2FilterAndEdge:
    def __init__(self,
                 img_path: str = "images/LowContrast.jpg",
                 saltpepper_prob: float = 0.05,
                 median_size: int = 3,
                 seed: int = 0):
        self.img_path = img_path
        self.p = float(saltpepper_prob)
        self.median_size = int(median_size)
        self.seed = int(seed)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img

    def _read_gray(self) -> np.ndarray:
        return self._to_gray(iio.imread(self.img_path))

    # Add salt-and-pepper noise
    def _add_salt_pepper(self, img: np.ndarray) -> np.ndarray:
        if self.p <= 0:
            return img.copy()
        rng = np.random.default_rng(self.seed)
        m = rng.random(img.shape)
        noisy = img.copy()
        noisy[m < self.p/2] = 0
        noisy[(m >= self.p/2) & (m < self.p)] = 255
        return noisy

    # Run the processing and display/save results
    def run(self, show: bool = True, save_dir: str | None = "outputs"):
        img = self._read_gray()
        noisy = self._add_salt_pepper(img)

        denoised = median_filter(noisy, size=self.median_size)
        mag_noisy, _ = calculate_gradient(noisy)
        mag_denoised, _ = calculate_gradient(denoised)

        print(f"salt&pepper p={self.p}, median size={self.median_size}")
        print("max |gradient| (noisy):    ", float(mag_noisy.max()))
        print("max |gradient| (denoised): ", float(mag_denoised.max()))

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0,0].imshow(noisy, cmap="gray");            axes[0,0].set_title("Noisy"); axes[0,0].axis("off")
        axes[0,1].imshow(denoised, cmap="gray");         axes[0,1].set_title("Median Filtered"); axes[0,1].axis("off")
        axes[1,0].imshow(mag_noisy, cmap="gray");        axes[1,0].set_title("Gradient Magnitude (Noisy)"); axes[1,0].axis("off")
        axes[1,1].imshow(mag_denoised, cmap="gray");     axes[1,1].set_title("Gradient Magnitude (After Median)"); axes[1,1].axis("off")
        fig.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "p2_summary.png"), dpi=200)
            print(f"[saved] ./"+os.path.join(save_dir, "p2_summary.png"))

        if show:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    Part2FilterAndEdge(
        img_path="images/LowContrast.jpg",
        saltpepper_prob=0.05,
        median_size=3,
        seed=0,
    ).run(show=True, save_dir="outputs")
