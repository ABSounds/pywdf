import numpy as np
from typing import Callable
from .wdf import baseWDF, rootWDF
from .rtype import RTypeAdaptor
from scipy.io import wavfile
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from scipy.fftpack import fft
import time


class Circuit:
    def __init__(self, source: baseWDF, root: rootWDF, output: baseWDF) -> None:
        """Initialize Circuit class functionality.

        Args:
            source (baseWDF): the circuit's voltage source
            root (rootWDF): the root of the wdf connection tree
            output (baseWDF): the component to be probed for output signal
        """
        self.source = source
        self.root = root
        self.output = output

    def process_sample(self, sample: float) -> float:
        """Process an individual sample with this circuit.

        Note: not every circuit will follow this general pattern, in such cases users may want to overwrite this function. See example circuits

        Args:
            sample (float): incoming sample to process

        Returns:
            float: processed sample
        """
        self.source.set_voltage(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.output.wave_to_voltage()

    def process_signal(self, signal: np.array) -> np.array:
        """Process an entire signal with this circuit.

        Args:
            signal (np.array): incoming signal to process

        Returns:
            np.array: processed signal
        """
        self.reset()
        return np.array([self.process_sample(sample) for sample in signal])

    def process_wav(self, filepath: str, output_filepath: str = None) -> np.array:
        fs, x = wavfile.read(filepath)
        if fs != self.fs:
            raise Exception(
                f"File sample rate differs from the {self.__class__.__name__}'s"
            )
        y = self.process_signal(x)
        if output_filepath is not None:
            wavfile.write(output_filepath, fs, y)
        return y

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0], "__iter__"):
            return self.process_signal(args[0])

    def get_impulse_response(self, delta_dur: float = 1, amp: float = 1) -> np.array:
        """Get circuit's impulse response

        Args:
            delta_dur (float, optional): duration of Dirac delta function in seconds. Defaults to 1.
            amp (float, optional): amplitude of delta signal's first sample. Defaults to 1.

        Returns:
            np.array: impulse response of the system
        """
        d = np.zeros(int(delta_dur * self.fs))
        d[0] = amp
        return self.process_signal(d)
    
    def plot_impulse_response(self, n_samples: int = 500, outpath: str = None, delta_dur: float = 1, amp: float = 1) -> None:
        plt.figure(figsize=(9, 5.85))
        plt.plot(self.get_impulse_response(delta_dur, amp)[:n_samples])
        plt.xlabel('Sample [n]')
        plt.ylabel('Amplitude [V]')
        plt.title(f'{self.__class__.__name__} impulse response')

        plt.grid()
        if outpath:
            plt.savefig(outpath)
        plt.show()

    def set_sample_rate(self, new_fs: float) -> None:
        """Change system's sample rate

        Args:
            new_fs (float): sample rate to change circuit to
        """
        if self.fs != new_fs:
            self.fs = new_fs
            for key in self.__dict__:
                if (
                    hasattr(self.__dict__[key], "fs")
                    and self.__dict__[key].fs != new_fs
                ):
                    self.__dict__[key].prepare(new_fs)

    def reset(self) -> None:
        """Return values of each circuit element's incident & reflected waves to 0"""
        for key in self.__dict__:
            if isinstance(self.__dict__[key], baseWDF):
                self.__dict__[key].reset()

    def compute_spectrum(self, fft_size: int = None) -> np.ndarray:
        x = self.get_impulse_response()
        N2 = int(fft_size / 2 - 1)
        H = fft(x, fft_size)[:N2]
        return H

    def plot_freqz(self, outpath: str = None, fft_size: int = None, xlim = [20, 20_000], xticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ylim: [int, int] = None, ystep: int = None, linewidth = 2):
        """Plot the circuit's frequency response

        Args:
            outpath (str, optional): filepath to save figure. Defaults to None.
            fft_size (int, optional): number of points of the FFT. Defaults to 2^15.
            xlim ([int, int], optional): [x_min, x_max], x axis limits for the plot. Defaults to [20, 20_000].
            ylim ([int, int], optional): [y_min, y_max], y axis limits for the plot. Defaults to [None, None].
            ystep (int, optional): space between the y axis marks. Defaults to None.
            linewidth (int, optional): width of the plot lines in pixels. Defaults to 2.
        """
        if fft_size is None:
            fft_size = int(2**15)
        H = self.compute_spectrum(fft_size)
        nyquist = self.fs / 2
        magnitude = 20 * np.log10(np.abs(H) + np.finfo(float).eps)
        phase = np.angle(H)
        N2 = int(fft_size / 2 - 1)
        frequencies = np.linspace(0, nyquist, N2)

        # TODO: improve frequency axis
        plt.rc('lines', linewidth = linewidth)
        ax = []
        # TODO: find a better y_size. This one's a bit chonky
        fig, ax1 = plt.subplots(ncols = 1, nrows=1, figsize = (10, 5))
        ax.append(ax1)
        xlims = [10**0, 10 ** np.log10(self.fs / 2)]
        ax[0].semilogx(frequencies, magnitude, color = "C00", linestyle = '-', label="Magnitude [dB]")
        ax[0].set_xlim(xlims)
        if ylim is None:
            #TODO: add variable offset (??)
            ymin = int(min(magnitude) - 2.0)
            ymax = int(max(magnitude) + 2.0)
            ylim = [ymin, ymax]
        ax[0].set_ylim(ylim)
        ax[0].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel("Magnitude [dBFs]")
        ax[0].grid(True, which = 'both')
        if ystep is None:
            ystep = np.ceil((ylim[1] - ylim[0]) / 10)
        ax[0].set_yticks (np.arange(ymin, ymax + ystep, step = ystep))
        ax.append(ax[0].twinx())
        
        phase = 180 * phase / np.pi
        ax[1].semilogx(frequencies, phase, color = "C01", linestyle = '--', label = "Phase [degrees]")
        ax[1].set_ylim([-180, 180])
        ax[1].set_yticks(np.arange(-180, 180, 30))
        ax[1].set_ylabel("Phase [degrees]")
        ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:0g}'.format(x)))
        ax[1].set_xticks(xticks)

        ax[0].set_title(
            loc="left", label=self.__class__.__name__
        )

        #TODO: I don't like the position of the legend using fig.legend() but ax.legend() overlaps the two axes legends
        fig.legend()
        plt.tight_layout()
        if outpath:
            plt.savefig(outpath)
        plt.show()
        
    def plot_freqz_list(
        self,
        values: list,
        set_function: Callable,
        param_label: str = "value",
        outpath: str = None,
    ):
        """Plot circuit's frequency response(s) while varying a parameter

        Args:
            values (list): list of parameter values to iterate through
            set_function (Callable): circuit's function for setting parameter
            param_label (str, optional): name of parameter being modulated. Defaults to 'value'.
            outpath (str, optional): filepath to save figure. Defaults to None.
        """
        # TODO: create a legend with values
        # TODO: add a label as input parameter to be use in the legend
        # TODO: split this function in compute_magnitude_and_phase
        # plot_magnitude, plot_phase. In that way we can reuse the methods in plot_freqz()
        fft_size = int(2**15)
        nyquist = self.fs / 2
        N2 = int(fft_size / 2 - 1)
        frequencies = np.linspace(0, nyquist, N2)

        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6.5))

        for value in values:
            print(f"{param_label} : {value}")
            set_function(value)
            x = self.get_impulse_response()

            h = fft(x, fft_size)[:N2]
            magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
            phase = np.angle(h)
            magnitude_peak = np.max(magnitude)
            top_offset = 10
            bottom_offset = 70

            xlims = [10**0, 10 ** np.log10(self.fs / 2)]
            ax[0].semilogx(frequencies, magnitude, label=f"{param_label} : {value}")
            ax[0].set_xlim(xlims)
            ax[0].set_ylim(
                [magnitude_peak - bottom_offset, magnitude_peak + top_offset]
            )
            ax[0].set_xlabel("Frequency [Hz]")
            ax[0].set_ylabel("Magnitude [dBFs]")
            ax[0].set_title(
                loc="left", label=self.__class__.__name__ + " magnitude response"
            )
            ax[0].grid(True)
            ax[0].legend()

            phase = 180 * phase / np.pi
            ax[1].semilogx(frequencies, phase, label=f"{param_label} : {value}")
            ax[1].set_xlim(xlims)
            ax[1].set_ylim([-180, 180])
            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel("Phase [degrees]")
            ax[1].set_title(
                loc="left", label=self.__class__.__name__ + " phase response"
            )
            ax[1].grid(True)
            ax[1].legend()

        plt.tight_layout()
        if outpath:
            plt.savefig(outpath)

        plt.show()

    def AC_transient_analysis(
        self,
        freq: float = 1000,
        amplitude: float = 1,
        t_ms: float = 5,
        outpath: str = None,
    ):
        """Plot transient analysis of Circuit's response to sine wave

        Args:
            freq (float, optional): frequency of sine wave. Defaults to 1000.
            amplitude (float, optional): amplitude of sine wave. Defaults to 1.
            t_ms (float, optional): time in ms of sine wave. Defaults to 5.
        """
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6.5))

        n_samples = int(t_ms * self.fs / 1000)

        n = np.arange(0, 2, 1 / self.fs)
        x = np.sin(2 * np.pi * freq * n) * amplitude
        y = self.process_signal(x)

        ax[0].plot(x[:n_samples], label="input signal")
        ax[0].plot(y[:n_samples], label="output signal", alpha=0.75)
        ax[0].set_xlabel("sample")
        ax[0].set_ylabel("amplitude")
        ax[0].set_title(loc="left", label="output signal vs input signal waveforms")
        ax[0].grid(True)
        ax[0].legend()

        N = len(n)
        w = scipy.signal.windows.hann((len(n)))
        y_fft = scipy.fft.fft(w * y)
        yf = scipy.fft.fftfreq(N, 1 / self.fs)[20 : N // 2]

        ax[1].plot(yf, 2.0 / N * np.abs(y_fft[20 : N // 2]))
        ax[1].set_xlabel("frequency [Hz]")
        ax[1].set_ylabel("magnitude")
        ax[1].set_title(loc="left", label="output signal spectrum")
        ax[1].grid(True)
        plt.tight_layout()
        if outpath:
            plt.savefig(outpath)

        plt.show()
    
    def measure_performance(self, t, n):
        """Measure performance of the circuit by processing n times a signal of length t seconds at self.fs sample rate.

        Args:
            t (float): time in seconds of the signal to process
            n (int): number of executions to average
        
        Returns:
            (float, float): (mean time taken to process t seconds of audio, real-time ratio)
        """
        times = []
        inputSignal = np.random.rand(int(t * self.fs))

        print(f'Time to process {t} seconds of audio with {self.__class__.__name__} at {self.fs / 1000:.0f} kHz:')
        for i in range(n):
            startTime = time.monotonic()
            self.process_signal(inputSignal)
            endTime = time.monotonic()
            totalTime = endTime - startTime
            times.append(totalTime)
            if (n != 1):
                print(f'  - Run {i+1} took {totalTime:.4}s')
        mean = np.mean(times)
        rt_ratio = t / mean
        print(f'\nAverage time taken to process {t} seconds of audio at {self.fs / 1000:.0f} kHz: {mean:.4}s')
        print(f'Average real-time ratio: {rt_ratio:.4}')
        return (mean, rt_ratio)
    
    def compare_with_LTspice(self, file_path, fs: float, n = 16384) -> None:
        """Plot the error between the WDF model and data exported from LTspice simulations

        You can export the data from LTspice by:
            1. Running a simulation with the following directives:
                .param fs = 48000
                .param n = 16384
                .ac lin {n} {fs/(2*n)} {fs/2}
            2. Exporting the data as a .txt file by selecting the plot window and then File -> Export data as text
            3. Make sure that the exporting format is set to Polar: (dB,deg) and you have selected the output voltage of the circuit

        Args:
            file_path (string): path to the .txt file exported from LTspice
            fs (float): sample rate used to export the data
            n (int, optional): number of points exported. Defaults to 16384
        """
        
        frequencies = []
        LTspice_magnitude_dB = []
        with open(file_path, 'r', encoding='latin-1') as file:
            next(file)
            for line in file:
                values = line.strip().split('\t')
                frequency = float(values[0])
                magnitude = float(values[1].split('(')[1].split('dB')[0])
                frequencies.append(frequency)
                LTspice_magnitude_dB.append(magnitude)
            frequencies = np.array(frequencies)
            LTspice_magnitude_dB = np.array(LTspice_magnitude_dB)

        self.set_sample_rate(fs)
        self.reset()
        pywdf_magnitude_linear = np.abs(self.compute_spectrum(fft_size=int((len(LTspice_magnitude_dB) + 2) * 2 + 1))[1:])
        pywdf_magnitude_dB = 20 * np.log10(pywdf_magnitude_linear + np.finfo(float).eps)
        error = (10 ** (LTspice_magnitude_dB / 20) - pywdf_magnitude_linear)**2

        _, ax = plt.subplots(figsize=(10, 4))
        plt.rc('lines', linewidth = 3)
        ax.semilogx(frequencies, LTspice_magnitude_dB, linestyle='-', color='C0', label=r"$|H_{LTSpice}(f)|$")
        ax.semilogx(frequencies, pywdf_magnitude_dB, linestyle='--', color='C1', label=r"$|H_{pywdf}(f)|$")
        ax.legend()
        ax.set_title(f'Magnitude response error at {int(self.fs/1000)} kHz sample rate')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_xlim(20, 20_000)
        ylim = (int(2.0 * np.round(np.min(LTspice_magnitude_dB - 2.0) / 2.0)), int(2.0 * np.round(np.max(LTspice_magnitude_dB + 2.0) / 2.0)))
        ax.set_ylim(ylim[0], ylim[1])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        error_matrix = np.tile(error, (3, 1))
        Y = np.linspace(ylim[0], ylim[1], num=3)
        norm = colors.LogNorm(vmin=1e-8, vmax=1e-3)
        pcm = ax.pcolormesh(frequencies, Y, error_matrix, cmap='binary', norm=norm)
        del error_matrix, Y
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label(r"$(|H_{\mathrm{LTSpice}}(f)| - |H_{\mathrm{pywdf}}(f)|)^2$")
        error_rms = np.sqrt(np.mean(error))
        latex_error_rms = r"${Error_{RMS}}|_{0 - fs/2}$" + f' = {error_rms:.3e}'
        ax.text(0.05, 0.1, latex_error_rms, bbox=dict(boxstyle='round', fc='w', ec='gray', alpha=0.7), transform=ax.transAxes)
        ax.grid(True, which='both')
        plt.tight_layout()
        plt.show()

    def _impedance_calc(self, R: RTypeAdaptor):
        """Placeholder function used to calculate impedance of Rtype adaptor

        Args:
            R (RTypeAdaptor): Rtype of which to calculate impedance
        """
        pass
