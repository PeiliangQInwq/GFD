"""Synthetic vibration signal generator for gearbox fault diagnosis.

This script produces a toy dataset of vibration signals representative of
gearbox fault conditions.  Each signal is simulated as a sum of sine
waves with different frequencies and phases, plus additive white
Gaussian noise.  Four classes are created: healthy, broken tooth,
wear and defect.  For convenience the script splits the data into
training and testing sets and saves them to ``dataset.npz`` under
``data/``.

Running this script will overwrite any existing ``dataset.npz`` in
the ``data`` folder.
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


def generate_signals(num_samples: int = 2000, sample_length: int = 1000,
                     sampling_rate: int = 1000) -> tuple:
    """Generate synthetic vibration signals for fault diagnosis.

    Args:
        num_samples: Total number of samples to generate (for all classes).
        sample_length: Number of points in each signal (duration = sample_length / sampling_rate seconds).
        sampling_rate: Sampling frequency in Hz.

    Returns:
        A tuple (X, y) where X is a NumPy array of shape (num_samples, sample_length)
        containing the signals and y is a NumPy array of class labels (0–3).
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # Number of classes
    num_classes = 4
    samples_per_class = num_samples // num_classes

    X = []
    y = []

    # Time vector
    t = np.arange(sample_length) / sampling_rate

    # Frequencies for different fault types
    freqs = [5.0, 20.0, 10.0, 15.0]  # Hz for healthy, broken tooth, wear, defect
    for class_idx, freq in enumerate(freqs):
        for _ in range(samples_per_class):
            # Base signal: sum of sine waves at the class frequency and its harmonics
            signal = np.sin(2 * np.pi * freq * t)
            if class_idx == 1:
                # Broken tooth: additional high frequency component
                signal += 0.5 * np.sin(2 * np.pi * (freq * 3) * t)
            elif class_idx == 2:
                # Wear: lower frequency plus first harmonic
                signal = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * (freq / 2) * t)
            elif class_idx == 3:
                # Defect: sum of two frequencies with phase shift
                signal = 0.8 * np.sin(2 * np.pi * freq * t + np.pi / 4) + 0.4 * np.sin(2 * np.pi * (freq * 1.5) * t)

            # Random amplitude modulation to simulate variability
            amp = 1.0 + 0.1 * rng.standard_normal()
            signal *= amp

            # Additive white Gaussian noise at random SNR between 0 and 20 dB
            snr_db = rng.uniform(0, 20)
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = rng.normal(scale=np.sqrt(noise_power), size=signal.shape)
            noisy_signal = signal + noise

            X.append(noisy_signal.astype(np.float32))
            y.append(class_idx)

    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic vibration data for gearbox fault diagnosis")
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples to generate in total')
    parser.add_argument('--length', type=int, default=1000, help='Number of points per sample')
    parser.add_argument('--sampling_rate', type=int, default=1000, help='Sampling frequency in Hz')
    args = parser.parse_args()

    X, y = generate_signals(num_samples=args.samples,
                            sample_length=args.length,
                            sampling_rate=args.sampling_rate)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Ensure the data folder exists
    data_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(data_dir, 'dataset.npz')
    np.savez(out_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Synthetic dataset saved to {out_path}")


if __name__ == '__main__':
    main()