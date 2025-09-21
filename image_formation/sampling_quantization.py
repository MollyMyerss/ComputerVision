import numpy as np
import matplotlib.pyplot as plt

signal_freq = 5.0   # Hz
duration    = 2.0   # seconds
sampling_freq = 8.0 # Hz
num_bits    = 3     # 3-bit quantization (8 levels: 0..7)
min_signal  = -1.0
max_signal  =  1.0

#continuous-time signal: sin(2*pi*f*t)
def original_signal(t, f=signal_freq):
    return np.sin(2 * np.pi * f * t)

#interval [0, duration) with n = fs*duration points
def sample_times(duration, fs):
    n = int(fs * duration)
    return np.linspace(0, duration, n, endpoint=False)

#uniform quantization in [xmin, xmax] to 2^nbits levels
def quantize_levels(x, nbits=num_bits, xmin=min_signal, xmax=max_signal):
    levels = (1 << nbits) 
    q_idx = np.rint((x - xmin) / (xmax - xmin) * (levels - 1)).astype(int)
    q_idx = np.clip(q_idx, 0, levels - 1)
    q_val = xmin + q_idx * (xmax - xmin) / (levels - 1)
    return q_idx, q_val

def main():
    t_dense = np.linspace(0, duration, 1000, endpoint=False)
    cont = original_signal(t_dense)

    t_s = sample_times(duration, sampling_freq)
    sampled = original_signal(t_s)

    _, q_vals = quantize_levels(sampled)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_dense, cont, label="Continuous signal")
    plt.step(t_s, q_vals, where="post",
             label=f"Quantized signal ({num_bits} bits)", linestyle="--")
    plt.scatter(t_s, sampled, s=18, label="Sampled points")

    plt.xlabel("Time (s)")
    plt.ylabel("Signal Value")
    plt.title("Sampling and Quantization")
    plt.grid(True)
    
    #move legend above plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=2, frameon=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
