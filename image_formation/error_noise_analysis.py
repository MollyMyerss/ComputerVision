import numpy as np
import matplotlib.pyplot as plt

#global parameters
signal_freq = 5.0 #sine-wave frequency (Hz)
duration    = 2.0 #seconds
sampling_freq = 8.0 #sampling rate (Hz)
num_bits    = 3 #3-bit quantization
min_signal  = -1.0 #quant min value
max_signal  =  1.0 #quant max value

noise_mean  = 0.0 #Gaussian noise mean
noise_std   = 0.1 

#Gaussian reference signal (sine @ frequenxy f)
def original_signal(t, f=signal_freq):
    return np.sin(2 * np.pi * f * t)

#interval over [0, duration)
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

#add Gaussian noise with given mean and std (as fraction of signal range)
def add_Gaussian_noise(signal, mean, std):
    mag = np.max(signal) - np.min(signal)
    noise = np.random.normal(mean, std * mag, len(signal))
    return signal + noise

#mean square error
def mse(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return np.mean((x - y) ** 2)

#root mean square error
def rmse(x, y):
    return np.sqrt(mse(x, y))

#peak signal-to-noise ratio
def psnr(x_ref, x_noisy, peak=None):
    m = mse(x_ref, x_noisy)
    if peak is None:
        peak = np.max(np.abs(x_ref))
    if m == 0:
        return float("inf")
    return 10.0 * np.log10((peak ** 2) / m)

def main():
    t_dense = np.linspace(0, duration, 2000, endpoint=False)
    cont = original_signal(t_dense)

    t_s = sample_times(duration, sampling_freq)
    sampled = original_signal(t_s)

    #sampled signal + Gaussian noise
    noisy_sampled = add_Gaussian_noise(sampled, noise_mean, noise_std)

    #quantize noisy samples
    _, quant_noisy = quantize_levels(noisy_sampled)

    #compute error metrics
    cont_at_samples = original_signal(t_s)
    mse_val  = mse(cont_at_samples, noisy_sampled)
    rmse_val = rmse(cont_at_samples, noisy_sampled)
    psnr_val = psnr(cont_at_samples, noisy_sampled, peak=np.max(np.abs(cont_at_samples)))

    #print MSE, RMSE, PSNR
    print("Mean square error, the root mean square error, and the peak signal-to-noise ratio:");
    print(f"  MSE  = {mse_val:.6f}")
    print(f"  RMSE = {rmse_val:.6f}")
    print(f"  PSNR = {psnr_val:.2f} dB")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_dense, cont, label="Continuous signal")
    plt.scatter(t_s, sampled, s=18, label="Sampled (clean)")
    plt.scatter(t_s, noisy_sampled, s=18, label="Sampled (noisy)")
    plt.step(t_s, quant_noisy, where="post", linestyle="--",
             label=f"Quantized noisy ({num_bits} bits)")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Value")
    plt.title("Noise & Error Analysis")
    plt.grid(True)

    #move legend above plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=2, frameon=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
