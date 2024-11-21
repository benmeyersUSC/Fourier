import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Constants
DURATION = 1  # in seconds
SAMPLE_RATE = 3000  # in Hz
RANGE = 100  # Frequency range
TIME_POINTS = np.arange(0, DURATION, 1 / SAMPLE_RATE)


def get_frequency(frequency: int, multiplier=None):
    if multiplier is None:
        amplitude = 1 + (np.random.random() * 0.5)
    else:
        amplitude = multiplier

    return amplitude * np.sin(2 * np.pi * int(frequency) * TIME_POINTS)


def get_frequency_matrix():
    """Generate a matrix of sine waves for Fourier transformation."""
    frequencies = [
        get_frequency(i) for i in range(1, RANGE + 1)
    ]
    return pd.DataFrame(frequencies)


class Signal:

    def __init__(self):
        self.signal = np.zeros_like(TIME_POINTS)
        self.frequencies = []

    def add_frequency(self, frequency: int, multiplier=None):
        if multiplier is None:
            amplitude = 1 + (np.random.random() * 0.5)
        else:
            amplitude = multiplier

        self.signal += amplitude * np.sin(2 * np.pi * int(frequency) * TIME_POINTS)
        self.frequencies.append((int(frequency), amplitude))

    def add_random_frequencies(self, count=5):
        for i in range(count):
            self.add_frequency(random.random() * RANGE)

    def set_gaussian_sound(self, length=SAMPLE_RATE*DURATION, std_dev=1):
        gaussian_length = length
        mean = gaussian_length // 2

        # Initialize array with zeros
        array = np.zeros(length)

        # Indices for the Gaussian curve
        start = (length - gaussian_length) // 2
        end = start + gaussian_length

        # Generate Gaussian curve
        x = np.arange(gaussian_length)
        gaussian = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

        # Place the Gaussian in the middle of the array
        array[start:end] = gaussian
        self.frequencies = []
        self.signal = array

    # def set_clap_sound(self, length=(SAMPLE_RATE * DURATION)):
    #     wv = list(np.zeros(length))
    #     wv[1] = 1
    #     self.frequencies = []
    #     self.signal = np.array(wv)
        # self.set_gaussian_sound(std_dev=1)

    # def set_clap_sound(self, length=(SAMPLE_RATE * DURATION)):
    #     # Create true impulse
    #     self.signal = np.zeros(length)
    #     middle_index = length // 2
    #     self.signal[middle_index] = 1.0
    #     self.frequencies = []

    def set_clap_sound(self, length=(SAMPLE_RATE * DURATION)):
        # Create a true impulse
        self.signal = np.zeros_like(TIME_POINTS)
        middle_index = len(TIME_POINTS) // 2
        self.signal[middle_index] = 1.0  # Single spike
        self.frequencies = []  # Clear any previous frequencies

    def plot(self):
        # Create figure and axes with specific height ratios
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [2, 1]})

        time_ms = TIME_POINTS * 1000

        # Plot main sound wave
        ax1.plot(time_ms, self.signal, label="Combined Sound Wave",
                 color='#1f77b4', linewidth=2.5)
        ax1.set_title("Synthetic Sound Wave", fontsize=14, pad=10)

        # Plot component frequencies
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.frequencies)))
        for freq, color in zip(self.frequencies, colors):
            ax2.plot(time_ms, get_frequency(freq[0]),
                     label=f"{freq[0]} Hz, strength: {freq[1]:.2f}", alpha=0.8,
                     linewidth=1.8)

        ax1.set_ylabel("Amplitude", fontsize=12)
        ax2.set_ylabel("Amplitude", fontsize=12)
        ax2.set_xlabel("Time (milliseconds)", fontsize=12)

        ax1.set_xticks(np.linspace(0, 1000, 11))
        ax2.set_xticks(np.linspace(0, 1000, 11))

        ax1.legend(fontsize=10, loc="upper right")
        ax2.legend(fontsize=10, title="Component Frequencies", loc="upper right")

        fig.suptitle("Sound Wave Composition", fontsize=16, y=0.95)

        plt.subplots_adjust(hspace=0.3)

        # plt.show()
        plt.savefig("plt.png")


class FourierTransform:
    def __init__(self, signal: Signal):
        self.signal = signal

        self.frequency_matrix = get_frequency_matrix()

    # def compute_frequencies(self):
    #     return self.signal.signal @ self.frequency_matrix.T
    # def compute_frequencies(self):
    #     # Add normalization factor
    #     dt = 1 / SAMPLE_RATE
    #     normalized_signal = self.signal.signal * dt
    #     return normalized_signal @ self.frequency_matrix.T / np.sqrt(2 * np.pi)
    #
    # def plot_computed_frequencies(self):
    #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    #
    #     computed_frequencies = self.compute_frequencies()
    #     cf = list(computed_frequencies)
    #     majors = []
    #
    #     mx = max(cf)
    #     while mx > 1000:
    #         idx = cf.index(mx)
    #         majors.append(idx+1)
    #         cf[idx] = 0
    #         mx = max(cf)
    #
    #     ax.plot(computed_frequencies, label=f"Fourier Transform Output => {majors}")
    #
    #     ax.legend(loc="upper right")
    #
    #     ax.set_xlabel("Frequency (Hz)")
    #     ax.set_ylabel("Frequency Strength")
    #
    #     fig.suptitle("Spectrum of Frequency Strength vs. Frequency in Input Signal")
    #     fig.tight_layout()
    #     fig.savefig("transform.png")
    #     # fig.show()

    def compute_frequencies(self):
        # Use numpy's FFT instead of manual matrix multiplication
        ft = np.fft.fft(self.signal.signal)
        # Get the magnitude spectrum
        magnitude_spectrum = np.abs(ft[:RANGE])  # Only take first RANGE frequencies
        # Normalize
        magnitude_spectrum = magnitude_spectrum / len(self.signal.signal)
        return magnitude_spectrum

    def plot_computed_frequencies(self):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        computed_frequencies = self.compute_frequencies()

        # Create frequency axis
        freq_axis = np.linspace(0, SAMPLE_RATE / 2, len(computed_frequencies))

        ax.plot(freq_axis, computed_frequencies,
                label="Fourier Transform Output")

        ax.legend(loc="upper right")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Frequency Strength")
        ax.grid(True, alpha=0.3)

        fig.suptitle("Spectrum of Frequency Strength vs. Frequency in Input Signal")
        fig.tight_layout()
        fig.savefig("transform.png")

    def _plot_freq_matrix(self):
        frequencies = [1, 10, 50, 100]

        # Create figure with specific height ratios (all equal in this case)
        fig, axes = plt.subplots(len(frequencies), 1, figsize=(15, 8), sharex=True)

        # Use a pleasing color palette
        colors = ['#2C3E50', '#E74C3C', '#27AE60', '#8E44AD']

        for i, freq in enumerate(frequencies):
            ax = axes[i]
            ax.plot(TIME_POINTS * 1000, self.frequency_matrix.iloc[freq - 1],
                    color=colors[i],
                    linewidth=1.5,
                    label=f'{freq} Hz')

            # Clean up each subplot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Remove y ticks since amplitude is normalized
            ax.set_yticks([])

            # Add minimal title and legend
            ax.set_title(f'{freq} Hz', loc='right', fontsize=10)

            # Set consistent y limits
            ax.set_ylim(-1.5, 1.5)

        # Only add x-label to bottom plot
        axes[-1].set_xlabel('Time (milliseconds)')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.1)
        plt.suptitle("Frequencies 1, 5, 50, 100 from Frequency Matrix")
        plt.tight_layout()
        plt.savefig("freq_mat.png")
        # plt.show()


if __name__ == "__main__":
    sound = Signal()
    sound.add_random_frequencies(count=9)
    # sound.set_clap_sound()
    sound.plot()

    ft = FourierTransform(sound)
    ft.plot_computed_frequencies()



