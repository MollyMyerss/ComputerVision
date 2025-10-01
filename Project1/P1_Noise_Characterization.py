import os, glob
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import pandas as pd

DARK_GLOB  = "images/dark/*.dng"
LIGHT_GLOB = "images/light/*.dng"
OUT_DIR    = "outputs"

def _get_raw_frame(rp):
    """Return a float32 RAW mosaic/visible plane from a rawpy object."""
    if hasattr(rp, "raw_image_visible") and rp.raw_image_visible is not None:
        arr = rp.raw_image_visible.astype(np.float32)
    else:
        arr = rp.raw_image.astype(np.float32)
    return arr

def _get_black_white(rp, arr):
    """Return (black_level, white_level) to normalize hist range."""
    try:
        blk = float(np.mean(rp.black_level_per_channel))
    except Exception:
        blk = float(np.percentile(arr, 0.1))
    try:
        wht = float(rp.white_level)
    except Exception:
        wht = float(np.percentile(arr, 99.9))
    if wht <= blk:
        wht = blk + max(1.0, float(arr.max() - blk))
    return blk, wht

def load_raw_gray(path, central_crop=0.1, normalize=False):
    """
    Load RAW with rawpy, return float32 array.
    - central_crop: fraction trimmed on each border (0.1 = keep central 80%).
    - normalize: if True, subtract black and clip to [0, white-black].
    """
    with rawpy.imread(path) as rp:
        raw = _get_raw_frame(rp)
        blk, wht = _get_black_white(rp, raw)
    if normalize:
        raw = np.clip(raw - blk, 0.0, wht - blk)
    h, w = raw.shape
    t = int(h * central_crop); b = int(h * (1.0 - central_crop))
    l = int(w * central_crop); r = int(w * (1.0 - central_crop))
    return raw[t:b, l:r]


def patch_stats(img, top=None, left=None, size=128):
    """Mean/std over a square patch; defaults to centered patch."""
    H, W = img.shape
    if top is None or left is None:
        top  = max(0, (H - size) // 2)
        left = max(0, (W - size) // 2)
    patch = img[top:top+size, left:left+size]
    return float(patch.mean()), float(patch.std())

def summarize(glob_pat, label, normalize=False):
    files = sorted(glob.glob(glob_pat))
    rows = []
    for fp in files:
        img = load_raw_gray(fp, central_crop=0.1, normalize=normalize)
        mu, sd = patch_stats(img, size=min(128, min(img.shape)//3))
        rows.append((os.path.basename(fp), mu, sd))
    print(f"\n=== {label} ({len(files)} files) ===")
    print("file, mean, std")
    for f, mu, sd in rows:
        print(f"{f}, {mu:.2f}, {sd:.2f}")
    return files, rows


def compute_hist_ranges(sample_path, dark_span=800.0, dark_pad=200.0):
    """
    Returns (rng_dark, rng_light) using RAW metadata.
    - rng_light: [black_level, white_level]
    - rng_dark : [black_level - pad, black_level + span] (coarse default)
    """
    with rawpy.imread(sample_path) as rp:
        arr = _get_raw_frame(rp).astype(np.float32)
        blk, wht = _get_black_white(rp, arr)
    rng_light = (max(0.0, float(blk)), float(wht))
    rng_dark  = (max(0.0, float(blk) - dark_pad), float(blk) + dark_span)
    return rng_dark, rng_light

# Plot histograms of multiple files on one plot
def plot_histograms(files, title, rng=None, bins=256, logy=False, density=False):
    plt.figure(figsize=(7,4))
    for fp in files:
        img = load_raw_gray(fp, central_crop=0.1, normalize=False)
        data = img.ravel()
        plt.hist(
            data, bins=bins, range=rng, histtype='step', linewidth=1,
            label=os.path.basename(fp), density=density
        )
    if rng is not None:
        plt.xlim(rng)                
    if logy:
        plt.yscale('log')
    plt.xlabel("RAW value")
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dark_files, dark_rows   = summarize(DARK_GLOB,  "Dark Frames",        normalize=False)
    light_files, light_rows = summarize(LIGHT_GLOB, "Bright/Flat Frames", normalize=False)

    if len(dark_files) == 0 or len(light_files) == 0:
        raise SystemExit("No files found. Check your paths:\n"
                         f"  DARK_GLOB  = {DARK_GLOB}\n"
                         f"  LIGHT_GLOB = {LIGHT_GLOB}")

    # Save stats
    pd.DataFrame(dark_rows,  columns=["file","mean","std"]).to_csv(os.path.join(OUT_DIR, "dark_stats.csv"),  index=False)
    pd.DataFrame(light_rows, columns=["file","mean","std"]).to_csv(os.path.join(OUT_DIR, "light_stats.csv"), index=False)

    # Ranges:
    rng_dark = (500.0, 600.0)  # fixed tight window for dark frames
    _, rng_light = compute_hist_ranges(light_files[0])  # metadata-driven for light

    print("\nHistogram ranges -> dark:", rng_dark, " light:", rng_light)

    # DARK: now using COUNTS (density=False). Keep log y if you like.
    plot_histograms(dark_files, "Dark-Frame Histograms",
                    rng=rng_dark, bins=1024, logy=True, density=False)
    plt.savefig(os.path.join(OUT_DIR, "dark_hist.png"), dpi=200); plt.clf()

    # LIGHT: counts (already), linear y
    plot_histograms(light_files, "Light/Flat Histograms",
                    rng=rng_light, bins=512, logy=False, density=False)
    plt.savefig(os.path.join(OUT_DIR, "bright_hist.png"), dpi=200); plt.clf()

    # console summary
    d_mu = np.mean([r[1] for r in dark_rows]);  d_sd = np.mean([r[2] for r in dark_rows])
    l_mu = np.mean([r[1] for r in light_rows]); l_sd = np.mean([r[2] for r in light_rows])
    print(f"\nAverages — Dark μ={d_mu:.1f}, σ={d_sd:.1f} | Bright μ={l_mu:.1f}, σ={l_sd:.1f}")
    print(f"\nArtifacts saved in ./{OUT_DIR}/ : dark_stats.csv, light_stats.csv, dark_hist.png, bright_hist.png\n")

if __name__ == "__main__":
    main()
