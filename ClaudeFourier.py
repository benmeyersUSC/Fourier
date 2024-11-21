import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
DURATION = 1.0  # seconds
SAMPLE_RATE = 3000  # Hz
NYQUIST_FREQ = SAMPLE_RATE // 2  # Maximum frequency we can analyze
TIME_POINTS = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
FREQ_POINTS = np.fft.fftfreq(len(TIME_POINTS), 1 / SAMPLE_RATE)


class Signal:
    def __init__(self):
        self.signal = np.zeros_like(TIME_POINTS)
        self.frequencies = []  # Store (frequency, amplitude) pairs
        self.duration = DURATION
        self.sample_rate = SAMPLE_RATE

    def add_frequency(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0):
        """
        Add a pure frequency component with specified amplitude and phase.

        Args:
            frequency: Frequency in Hz
            amplitude: Wave amplitude
            phase: Phase offset in radians
        """
        self.signal += amplitude * np.sin(2 * np.pi * frequency * TIME_POINTS + phase)
        self.frequencies.append((frequency, amplitude, phase))

    def add_random_frequencies(self, count: int = 5, max_freq: float = None):
        """Add random frequency components."""
        if max_freq is None:
            max_freq = NYQUIST_FREQ

        for _ in range(count):
            freq = np.random.uniform(0, max_freq)
            amp = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            self.add_frequency(freq, amp, phase)

    def set_gaussian_pulse(self, center: float = None, std_dev: float = 0.01):
        """
        Create a Gaussian pulse centered at specified time.

        Args:
            center: Center time in seconds
            std_dev: Standard deviation in seconds
        """
        if center is None:
            center = DURATION / 2

        # Convert time to array indices
        center_idx = int(center * SAMPLE_RATE)
        std_dev_pts = std_dev * SAMPLE_RATE

        # Generate Gaussian
        x = np.arange(len(TIME_POINTS))
        self.signal = np.exp(-0.5 * ((x - center_idx) / std_dev_pts) ** 2)
        self.signal /= np.sum(self.signal)  # Normalize
        self.frequencies = []

    def set_delta_function(self, center: float = None):
        """
        Create a discrete approximation of a delta function.

        Args:
            center: Center time in seconds
        """
        if center is None:
            center = DURATION / 2

        self.signal = np.zeros_like(TIME_POINTS)
        center_idx = int(center * SAMPLE_RATE)
        self.signal[center_idx] = 1.0
        self.frequencies = []

    def plot_signal(self, title: str = "Signal in Time Domain"):
        """Plot the signal in time domain."""
        plt.figure(figsize=(15, 6))
        plt.plot(TIME_POINTS * 1000, self.signal, 'b-', label='Signal')
        plt.title(title)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("signal.png")
        plt.close()


class FourierTransform:
    def __init__(self, signal: Signal):
        self.signal = signal
        self.spectrum = None
        self.frequencies = None
        self.compute_transform()

    def compute_transform(self):
        """
        Compute the Fourier transform using FFT.
        Properly normalized for quantum mechanical interpretation.
        """
        # Compute FFT and shift zero frequency to center
        fft_raw = np.fft.fft(self.signal.signal)
        self.spectrum = np.fft.fftshift(fft_raw)

        # Get corresponding frequencies
        self.frequencies = np.fft.fftshift(FREQ_POINTS)

        # Normalize for quantum interpretation
        dt = 1 / SAMPLE_RATE
        self.spectrum *= dt / np.sqrt(2 * np.pi)

    def get_magnitude_spectrum(self):
        """Get the magnitude spectrum."""
        return np.abs(self.spectrum)

    def get_phase_spectrum(self):
        """Get the phase spectrum."""
        return np.angle(self.spectrum)

    def get_power_spectrum(self):
        """Get the power spectrum."""
        return np.abs(self.spectrum) ** 2

    def plot_spectra(self, max_freq: float = None):
        """
        Plot magnitude, phase, and power spectra.

        Args:
            max_freq: Maximum frequency to plot (Hz)
        """
        if max_freq is None:
            max_freq = NYQUIST_FREQ

        # Create mask for frequency range
        freq_mask = np.abs(self.frequencies) <= max_freq

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Magnitude spectrum
        ax1.plot(self.frequencies[freq_mask], self.get_magnitude_spectrum()[freq_mask])
        ax1.set_title('Magnitude Spectrum')
        ax1.set_ylabel('|F(ω)|')
        ax1.grid(True, alpha=0.3)

        # Phase spectrum
        ax2.plot(self.frequencies[freq_mask], self.get_phase_spectrum()[freq_mask])
        ax2.set_title('Phase Spectrum')
        ax2.set_ylabel('∠F(ω) (radians)')
        ax2.grid(True, alpha=0.3)

        # Power spectrum
        ax3.plot(self.frequencies[freq_mask], self.get_power_spectrum()[freq_mask])
        ax3.set_title('Power Spectrum')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('|F(ω)|²')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("spectra.png")
        plt.close()


def demonstrate_uncertainty_principle():
    """Demonstrate the time-frequency uncertainty principle."""
    # Create signals with different time localization
    signals = []
    std_devs = [0.001, 0.01, 0.05, 0.1]  # Different width Gaussians

    for std in std_devs:
        sig = Signal()
        sig.set_gaussian_pulse(std_dev=std)
        signals.append((sig, std))

    # Plot time and frequency representations
    fig, axs = plt.subplots(len(signals), 2, figsize=(15, 4 * len(signals)))

    for i, (sig, std) in enumerate(signals):
        # Time domain
        axs[i, 0].plot(TIME_POINTS * 1000, sig.signal)
        axs[i, 0].set_title(f'Gaussian Pulse (σ = {std * 1000:.1f} ms)')
        axs[i, 0].set_xlabel('Time (ms)')
        axs[i, 0].set_ylabel('Amplitude')

        # Frequency domain
        ft = FourierTransform(sig)
        freq_mask = np.abs(ft.frequencies) <= 100  # Plot up to 100 Hz
        axs[i, 1].plot(ft.frequencies[freq_mask], ft.get_magnitude_spectrum()[freq_mask])
        axs[i, 1].set_title(f'Frequency Spectrum (σ = {std * 1000:.1f} ms)')
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_ylabel('Magnitude')

    plt.tight_layout()
    plt.savefig("uncertainty.png")
    plt.close()


if __name__ == "__main__":
    # # Example 1: Demonstrate uncertainty principle
    # demonstrate_uncertainty_principle()
    #
    # # Example 2: Create and analyze a delta function
    # signal = Signal()
    # signal.set_delta_function()
    # signal.plot_signal("Delta Function Approximation")
    #
    # ft = FourierTransform(signal)
    # ft.plot_spectra(max_freq=100)
    #
    # # Example 3: Analyze a complex signal
    # complex_signal = Signal()
    # complex_signal.add_random_frequencies(count=5, max_freq=50)
    # complex_signal.plot_signal("Complex Signal with Random Frequencies")
    #
    # ft_complex = FourierTransform(complex_signal)
    # ft_complex.plot_spectra(max_freq=100)

    # [Previous code remains exactly the same until the if __name__ == "__main__": section]

    def demonstrate_all_cases():
        """
        Comprehensive demonstration of three key cases:
        1. Random wave composition
        2. Delta function (clap)
        3. Multiple Gaussians of varying widths
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Case 1: Random Wave
        signal_random = Signal()
        # Add specific frequencies for clearer visualization
        signal_random.add_frequency(10, 1.0, 0)  # 10 Hz
        signal_random.add_frequency(25, 0.5, np.pi / 4)  # 25 Hz
        signal_random.add_frequency(40, 0.7, np.pi / 3)  # 40 Hz

        ax1_time = fig.add_subplot(gs[0, 0])
        ax1_freq = fig.add_subplot(gs[0, 1])

        ax1_time.plot(TIME_POINTS * 1000, signal_random.signal, 'b-', label='Composite Signal')
        ax1_time.set_title('Random Wave Composition (Time Domain)')
        ax1_time.set_xlabel('Time (ms)')
        ax1_time.set_ylabel('Amplitude')
        ax1_time.grid(True, alpha=0.3)
        ax1_time.legend()

        ft_random = FourierTransform(signal_random)
        freq_mask = np.abs(ft_random.frequencies) <= 50  # Show up to 50 Hz
        ax1_freq.plot(ft_random.frequencies[freq_mask],
                      ft_random.get_magnitude_spectrum()[freq_mask],
                      'b-', label='Frequency Spectrum')
        ax1_freq.set_title('Random Wave Spectrum')
        ax1_freq.set_xlabel('Frequency (Hz)')
        ax1_freq.set_ylabel('Magnitude')
        ax1_freq.grid(True, alpha=0.3)
        ax1_freq.legend()

        # Case 2: Delta Function (Clap)
        signal_delta = Signal()
        signal_delta.set_delta_function()

        ax2_time = fig.add_subplot(gs[1, 0])
        ax2_freq = fig.add_subplot(gs[1, 1])

        ax2_time.plot(TIME_POINTS * 1000, signal_delta.signal, 'r-', label='Delta Function')
        ax2_time.set_title('Delta Function / Clap (Time Domain)')
        ax2_time.set_xlabel('Time (ms)')
        ax2_time.set_ylabel('Amplitude')
        ax2_time.grid(True, alpha=0.3)
        ax2_time.legend()

        ft_delta = FourierTransform(signal_delta)
        ax2_freq.plot(ft_delta.frequencies[freq_mask],
                      ft_delta.get_magnitude_spectrum()[freq_mask],
                      'r-', label='Frequency Spectrum')
        ax2_freq.set_title('Delta Function Spectrum')
        ax2_freq.set_xlabel('Frequency (Hz)')
        ax2_freq.set_ylabel('Magnitude')
        ax2_freq.grid(True, alpha=0.3)
        ax2_freq.legend()

        # Case 3: Multiple Gaussians
        ax3_time = fig.add_subplot(gs[2, 0])
        ax3_freq = fig.add_subplot(gs[2, 1])

        std_devs = [0.005, 0.02, 0.05]  # Different width Gaussians
        colors = ['g', 'c', 'm']

        for std, color in zip(std_devs, colors):
            signal_gauss = Signal()
            signal_gauss.set_gaussian_pulse(std_dev=std)

            ax3_time.plot(TIME_POINTS * 1000, signal_gauss.signal,
                          color=color, label=f'σ = {std * 1000:.1f} ms')

            ft_gauss = FourierTransform(signal_gauss)
            ax3_freq.plot(ft_gauss.frequencies[freq_mask],
                          ft_gauss.get_magnitude_spectrum()[freq_mask],
                          color=color, label=f'σ = {std * 1000:.1f} ms')

        ax3_time.set_title('Gaussian Pulses (Time Domain)')
        ax3_time.set_xlabel('Time (ms)')
        ax3_time.set_ylabel('Amplitude')
        ax3_time.grid(True, alpha=0.3)
        ax3_time.legend()

        ax3_freq.set_title('Gaussian Pulse Spectra')
        ax3_freq.set_xlabel('Frequency (Hz)')
        ax3_freq.set_ylabel('Magnitude')
        ax3_freq.grid(True, alpha=0.3)
        ax3_freq.legend()

        plt.suptitle('Time-Frequency Domain Relationships\nDemonstrating the Uncertainty Principle',
                     fontsize=16, y=0.95)
        plt.savefig("comprehensive_demo.png")
        plt.close()

    demonstrate_all_cases()

