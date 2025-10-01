import sys
import math
import argparse
from pathlib import Path

print("\n=== Part 2 Runner ===")
print(f"Python: {sys.executable}")

missing = []
try:
    import numpy as np
except Exception: missing.append("numpy")
try:
    import rawpy
    HAVE_RAWPY = True
except Exception:
    HAVE_RAWPY = False
try:
    import pandas as pd
except Exception: missing.append("pandas")
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
except Exception: missing.append("matplotlib")
try:
    import exifread   
    HAVE_EXIFREAD = True
except Exception:
    HAVE_EXIFREAD = False
try:
    import tifffile   
    HAVE_TIFFFILE = True
except Exception:
    HAVE_TIFFFILE = False
try:
    from PIL import Image  
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

if missing:
    print(f"\nERROR: Missing required packages: {', '.join(missing)}")
    print("Install:\n  pip3 install " + " ".join(missing))
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR  = SCRIPT_DIR / "outputs_part2"
PATCH_SIZE = 128
CENTER_CROP_FRAC = 0.10

MAKE_NORM = True

SENSOR_WIDTH_MM_FALLBACK = {"main": None, "tele": None}  # e.g., {"main": 7.0, "tele": 6.4}

p = argparse.ArgumentParser(description="Part 2: Investigating Multi-Camera Systems")
p.add_argument("--main", type=str, required=True, help="Path to Normal (main) image/DNG")
p.add_argument("--tele", type=str, required=True, help="Path to Zoom (telephoto) image/DNG")
args = p.parse_args()

MAIN_PATH = Path(args.main).expanduser().resolve()
TELE_PATH = Path(args.tele).expanduser().resolve()

print(f"CWD   : {Path.cwd()}")
print(f"File  : {Path(__file__).resolve()}")
print(f"Main  : {MAIN_PATH}")
print(f"Tele  : {TELE_PATH}")

for pth in [MAIN_PATH, TELE_PATH]:
    if not pth.exists():
        print("ERROR: path not found:", pth); sys.exit(1)

def read_exif(path: Path):
    if not HAVE_EXIFREAD: return {}
    with open(path, "rb") as f:
        return exifread.process_file(f, details=False)

def exif_focal_length_mm(tags: dict):
    tag = tags.get("EXIF FocalLength")
    if not tag: return None
    v = getattr(tag, "values", tag)
    try:
        return float(v[0].num) / float(v[0].den)
    except Exception:
        try: return float(str(tag))
        except Exception: return None

def exif_sensor_width_mm(path: Path, tags: dict):
    width_px = None
    for key in ("EXIF ExifImageWidth", "Image ImageWidth"):
        t = tags.get(key)
        if t:
            try: width_px = int(str(t)); break
            except Exception: pass
    if width_px is None:
        t = tags.get("Image DefaultCropSize")
        if t:
            try: width_px = int(t.values[0])
            except Exception: pass
    xres = tags.get("EXIF FocalPlaneXResolution")
    unit = tags.get("EXIF FocalPlaneResolutionUnit")  
    if width_px and xres and unit:
        try:
            ratio = xres.values[0]
            x_per_unit = float(ratio.num)/float(ratio.den)
            unit_code = int(str(unit))
            if unit_code == 2: pixels_per_mm = x_per_unit / 25.4
            elif unit_code == 3: pixels_per_mm = x_per_unit / 10.0
            else: pixels_per_mm = None
            if pixels_per_mm and pixels_per_mm > 0:
                return width_px / pixels_per_mm
        except Exception:
            pass
    return None

def _central_crop(img: np.ndarray, frac=CENTER_CROP_FRAC):
    H, W = img.shape[:2]
    t = int(H*frac); b = int(H*(1.0-frac))
    l = int(W*frac); r = int(W*(1.0-frac))
    return img[t:b, l:r]

def load_raw_or_image(path: Path):
    """
    Returns:
      kind: "raw_mosaic" | "gray_from_tiff" | "gray_from_rgb"
      img: 2D float32 array
      blk, wht: black/white levels if known, else (percentiles)
      note: which loader was used
    """
    if HAVE_RAWPY:
        try:
            with rawpy.imread(str(path)) as rp:
                if hasattr(rp, "raw_image_visible") and rp.raw_image_visible is not None:
                    arr = rp.raw_image_visible.astype(np.float32)
                else:
                    arr = rp.raw_image.astype(np.float32)
                try: blk = float(np.mean(rp.black_level_per_channel))
                except Exception: blk = float(np.percentile(arr, 0.1))
                try: wht = float(rp.white_level)
                except Exception: wht = float(np.percentile(arr, 99.9))
                if wht <= blk:
                    wht = blk + max(1.0, float(arr.max()-blk))
                return "raw_mosaic", _central_crop(arr), blk, wht, "rawpy"
        except Exception as e:
            print(f"[info] rawpy failed on {path.name}: {e}")

    if HAVE_TIFFFILE:
        try:
            data = tifffile.imread(str(path))
            if data.ndim == 2:
                img = data.astype(np.float32)
                blk = float(np.percentile(img, 0.1))
                wht = float(np.percentile(img, 99.9))
                return "gray_from_tiff", _central_crop(img), blk, wht, "tifffile"
            else:
                arr = data.astype(np.float32)
                if arr.shape[2] >= 3:
                    r,g,b = arr[...,0], arr[...,1], arr[...,2]
                    gray = 0.2126*r + 0.7152*g + 0.0722*b
                else:
                    gray = arr.mean(axis=2)
                blk = float(np.percentile(gray, 0.1))
                wht = float(np.percentile(gray, 99.9))
                return "gray_from_tiff", _central_crop(gray), blk, wht, "tifffile"
        except Exception as e:
            print(f"[info] tifffile failed on {path.name}: {e}")

    if HAVE_PIL:
        try:
            with Image.open(str(path)) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32)
                r,g,b = arr[...,0], arr[...,1], arr[...,2]
                gray = 0.2126*r + 0.7152*g + 0.0722*b
                gray = _central_crop(gray)
                blk = float(np.percentile(gray, 0.1))
                wht = float(np.percentile(gray, 99.9))
                return "gray_from_rgb", gray, blk, wht, "PIL"
        except Exception as e:
            print(f"[info] PIL failed on {path.name}: {e}")

    raise SystemExit(f"Cannot read image: {path}")

def patch_stats(img: np.ndarray, size=PATCH_SIZE):
    H, W = img.shape
    top  = max(0, (H - size)//2)
    left = max(0, (W - size)//2)
    patch = img[top:top+size, left:left+size]
    return float(patch.mean()), float(patch.std())

def plot_histogram_counts(img: np.ndarray, title: str, out_path: Path, rng=None, bins=512):
    data = img.ravel()
    plt.figure(figsize=(7,4))
    plt.hist(data, bins=bins, range=rng, histtype='step', linewidth=1, density=False)
    if rng: plt.xlim(rng)
    plt.title(title); plt.xlabel("Value"); plt.ylabel("Count")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))  
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180); plt.close()

def horizontal_fov_deg(sensor_width_mm: float, focal_length_mm: float) -> float:
    return 2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * focal_length_mm)))

def analyze_one(label_key: str, path: Path, fallback_sensor_mm):
    label = "Main" if label_key == "main" else "Telephoto"

    tags = read_exif(path)
    focal_mm = exif_focal_length_mm(tags)
    sensor_w_mm = exif_sensor_width_mm(path, tags) or fallback_sensor_mm

    kind, img, blk, wht, loader = load_raw_or_image(path)

    img_norm = np.clip(img - blk, 0.0, max(1.0, wht - blk))

    mu, sd = patch_stats(img_norm, size=PATCH_SIZE)

    fov = None
    if sensor_w_mm and focal_mm:
        fov = horizontal_fov_deg(sensor_w_mm, focal_mm)

    if MAKE_NORM:
        base = path.stem
        xmax = min(8000, max(1000.0, wht - blk))
        plot_histogram_counts(
            img_norm, f"{label} Normalized Histogram",
            OUT_DIR / f"{base}_norm_hist.png",
            rng=(0, xmax), bins=512
        )

    note = []
    if not HAVE_EXIFREAD: note.append("exifread not installed")
    if sensor_w_mm is None: note.append("sensor_width_mm missing (set fallback)")
    return {
        "label": label,
        "file": path.name,
        "loader": loader,
        "kind": kind,
        "raw_black_level": blk,
        "raw_white_level": wht,
        "noise_mu_patch": mu,
        "noise_sigma_patch": sd,
        "focal_length_mm": focal_mm,
        "sensor_width_mm": sensor_w_mm,
        "h_fov_deg": fov,
        "note": "; ".join(note) if note else "ok"
    }

def main():
    rows = []
    rows.append(analyze_one("main", MAIN_PATH, SENSOR_WIDTH_MM_FALLBACK["main"]))
    rows.append(analyze_one("tele", TELE_PATH, SENSOR_WIDTH_MM_FALLBACK["tele"]))

    df = pd.DataFrame(rows, columns=[
        "label","file","loader","kind",
        "raw_black_level","raw_white_level",
        "noise_mu_patch","noise_sigma_patch",
        "focal_length_mm","sensor_width_mm","h_fov_deg","note"
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "multi_camera_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved summary to: {out_csv}")
    print(f"Artifacts saved to: {OUT_DIR}/ (only *_norm_hist.png files)\n")

if __name__ == "__main__":
    main()
()