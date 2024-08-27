import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import freqz, lti
from scipy.fft import fftshift, fft

from eyediagram.mpl import eyediagram


def plot_bar(labels: list, heights: list, axes: plt.Axes):
    assert len(labels) == len(heights)

    # Determine the number of bars
    num_bars = len(labels)

    # Create the bar plot
    axes.bar(range(num_bars), heights)

    # Set the tick positions and labels
    axes.set_xticks(range(num_bars))
    axes.set_xticklabels(labels)

    # Rotate the x-axis labels if needed
    if num_bars > 10:
        plt.xticks(rotation=45, ha='right')


def plot_fft_filter_response(h: npt.ArrayLike, ax: plt.Axes, Ts: float,
                            plot_label=None, n_fft=2048, color=None, linestyle=None):
    # Do freqz of the filter and redefine fq axis (double sided)
    fqs_freqz, H = freqz(h, 1, fs=1/Ts, whole=True, worN=n_fft)
    fqs_freqz = np.arange(-len(fqs_freqz)//2, len(fqs_freqz)//2) / (len(fqs_freqz) * Ts)

    # Plot ampltidue response in dB domain
    ax.plot(fqs_freqz, 20.0 * np.log10(fftshift(np.absolute(H))), label=plot_label,
            color=color, linestyle=linestyle)
    ax.grid(True)


def plot_fft_ab_response(fb: npt.ArrayLike, fa: npt.ArrayLike,
                         ax: plt.Axes, Ts: float, plot_label=None, n_fft=2048):
    eps = 1e-16

    # Do freqz of the filter and redefine fq axis (double sided)
    fqs_freqz, H = freqz(fb, fa, fs=1/Ts, whole=True, worN=n_fft)
    fqs_freqz = np.arange(-len(fqs_freqz)//2, len(fqs_freqz)//2) / (len(fqs_freqz) * Ts)

    # Plot ampltidue response in dB domain
    ax.plot(fqs_freqz, 20.0 * np.log10(fftshift(np.absolute(H)) + eps), label=plot_label)
    ax.grid(True)

def plot_fft(x: npt.ArrayLike, ax: plt.Axes, Ts: float,
             plot_label=None):
    eps = 1e-16

    N_fft = len(x)
    H = fft(x) * Ts
    fqs = np.arange(-N_fft//2, N_fft//2) / (N_fft * Ts)
    Hshifted = fftshift(H)

    ax.plot(fqs, 20.0 * np.log10(np.absolute(Hshifted) + eps), label=plot_label)
    ax.grid(True)


def plot_eyediagram(rx_out: npt.ArrayLike, ax: plt.Axes, Ts: float, sps: int, histogram: bool = False,
                    decimation=10, n_symbol_periods=4, shift=0):
    t = np.arange(shift * Ts, shift * Ts + n_symbol_periods * Ts * sps, Ts)
    discard_n_symbol_periods = 10
    if histogram:
        grid_res = (8, 16)
        eyediagram(np.roll(rx_out, shift), ax=ax, window_size=n_symbol_periods * sps, offset=shift, colorbar=False,
                   bins=(grid_res[0] * n_symbol_periods * sps, grid_res[1] * n_symbol_periods * sps),
                   cmap='Reds')
        ax.set_xticks(np.arange(sps * grid_res[1], n_symbol_periods * sps * grid_res[1], sps * grid_res[1]))
        ax.set_xticklabels([f"{i} / Rs" for i  in np.arange(1, n_symbol_periods)])
        
        # Eyediagram is plotted by default in y-range (ymin - 0.05 * A, ymax + 0.05 * A)
        yamp = rx_out.max() - rx_out.min()
        yvals = np.linspace(rx_out.min() - 0.05 * yamp, rx_out.max() + 0.05 * yamp, grid_res[0] * n_symbol_periods * sps)
        n_yvals = 4
        # Find the indices of elements closest to integers within the tolerance
        intindices = np.where(yvals - np.round(yvals) <= 0.01)[0]
        new_yticks = intindices[len(intindices)//n_yvals::len(intindices)//n_yvals]
        new_ytickvalues = np.take(np.flip(yvals), new_yticks)
        ax.set_yticks(new_yticks)
        ax.set_yticklabels([f"{yv:.0f}" for yv in new_ytickvalues])
    else:
        ax.plot(t, np.reshape(np.roll(rx_out, shift), (-1, sps * n_symbol_periods))[discard_n_symbol_periods:-discard_n_symbol_periods:decimation].T,
                color='crimson', alpha=.1, lw=.5)
        ax.grid(True)
        ax.set_xlim((np.min(t), np.max(t)))


def plot_pole_zero(system_transfer_function: tuple, ax: plt.Axes):
    lti_sys = lti(*system_transfer_function)
    print(f"System has {len(lti_sys.poles)} poles and {len(lti_sys.zeros)} zeros")

    # Plot unit circle
    rad = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(rad), np.sin(rad), 'k-')

    # Plot poles
    ax.plot(lti_sys.poles.real, lti_sys.poles.imag, 'rx')
    
    # Plot zeros
    ax.plot(lti_sys.zeros.real, lti_sys.zeros.imag, 'ro')

    ax.grid(True)
    # ax.axis('equal')
