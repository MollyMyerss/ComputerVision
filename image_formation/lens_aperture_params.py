import numpy as np
import matplotlib.pyplot as plt

# Thin-lens: 1/f = 1/z0 + 1/zi so zi = f z0 / (z0 - f)
def thin_lens_zi(f, z0):
    z0 = np.asarray(z0, dtype=float)
    return (f * z0) / (z0 - f)

def plot_zi_vs_z0(f_list):
    plt.figure(figsize=(8, 6))
    for f in f_list:
        z0_min = 1.1 * f          
        # 4 points per mm over [1.1 f, 1e4] mm 
        n_pts = max(1000, int((1e4 - z0_min) * 4))
        z0 = np.linspace(z0_min, 1e4, n_pts)
        zi = thin_lens_zi(f, z0)
        # draw curve and reuse its color for the line at z0 = f
        [line] = plt.loglog(z0, zi, label=f"f = {f} mm")
        # vertical dashed line at z0 = f
        plt.axvline(f, linestyle="--", color=line.get_color(), alpha=0.6)

    plt.ylim(1, 3000)
    plt.xlabel("Object distance z0 (mm)")
    plt.ylabel("Image distance zi (mm)")
    plt.title("Thin Lens: zi vs z0 for 4 different focal lengths")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_aperture_vs_f():
    import matplotlib.pyplot as plt

    # Lines D = f/N
    f_vals = np.linspace(3, 600, 500)
    f_numbers = [1.4, 1.8, 2.8, 4.0, 8.0]

    fig, ax = plt.subplots(figsize=(8, 6))
    for N in f_numbers:
        ax.plot(f_vals, f_vals / N, label=f"f/{N}")

    # Lens points to mark
    lenses = [
        (24, 1.4),
        (50, 1.8),
        ((70, 200), 2.8),
        (400, 2.8),
        (600, 4.0),
    ]

    #point creation
    points = []
    for f_entry, N in lenses:
        if isinstance(f_entry, tuple):
            fmin, fmax = f_entry
            points.extend([(fmin, fmin / N), (fmax, fmax / N)])
        else:
            points.append((f_entry, f_entry / N))

    xs, ys = zip(*points)
    for x, y in points: ax.plot(x, y, '.', ms=8, zorder=5)

    ax.set_xlabel("Focal length f (mm)")
    ax.set_ylabel("Aperture diameter D (mm)")
    ax.set_title("Aperture Diameter vs Focal Length")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_required_diameters():
    # (focal length, max f-number)-- (min,max)
    lenses = [
        (24, 1.4),
        (50, 1.8),
        ((70, 200), 2.8),
        (400, 2.8),
        (600, 4.0),
    ]
    
    print("Aperture diameters for plotted f-numbers:")
    # D = f / N
    for f_entry, N in lenses:
        if isinstance(f_entry, tuple):
            fmin, fmax = f_entry
            print(f"  {fmin}-{fmax} mm at f/{N}: D ≈ {fmin/N:.2f}–{fmax/N:.2f} mm")
        else:
            print(f"  {f_entry} mm at f/{N}: D ≈ {f_entry/N:.2f} mm")

def main():
    f_list = [3, 9, 50, 200]  # mm per spec
    plot_zi_vs_z0(f_list)
    print_required_diameters()
    plot_aperture_vs_f()

if __name__ == "__main__":
    main()
