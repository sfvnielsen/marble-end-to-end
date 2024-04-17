"""
    Script that runs end-to-end learning, calculates SER and plots system response
"""

import os
import komm
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from lib.utility import calc_ser_pam, calc_theory_ser_pam
from lib.systems import IntensityModulationChannel
from lib.plotting import plot_bar, plot_fft_filter_response, plot_fft_ab_response, plot_eyediagram

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 20}

text = {'usetex': True}

mpl.rc('font', **font)
mpl.rc('text', **text)

FIGSIZE = (12.5, 7.5)
DPI = 150
FIGURE_DIR = 'figures'
FIGPREFIX = 'e2e_imdd'


if __name__ == "__main__":
    # Define simulation parameters
    save_figures = False
    n_symbols_train = int(1e6)
    n_symbols_val = int(1e6)  # number of symbols used for SER calculation
    samples_per_symbol = 4
    baud_rate = int(100e9)
    mod_order = 4  # PAM
    rrc_rolloff = 0.01
    learn_tx, tx_filter_length = True, 35
    learn_rx, rx_filter_length = True, 35
    dac_bwl_relative_cutoff = 0.9  # low-pass filter cuttoff relative to information bandwidth
    adc_bwl_relative_cutoff = 0.9
    use_1clr = True

    # Configuration of electro absorption modulator
    ideal_modulator = True
    eam_config = {
        'insertion_loss': 0.0,
        'pp_voltage': 3.0,
        'bias_voltage': -1.5,
        'laser_power_dbm': -5.0,
        'linear_absorption': True
    }

    # Channel configuration - single model fiber
    smf_config = {
        'fiber_length': 2.0,
        'attenuation': 0.0,
        'carrier_wavelength': 1270,
        'zero_dispersion_wavelength': 1310,
        'dispersion_slope': 0.092
        # 'dispersion_parameter': 16.0
    }

    # Configuration of the photodiode
    photodiode_config = {
        'responsivity': 1.0,
        'temperature': 20.0,
        'dark_current': 10e-9,
        'impedance_load': 50.0
    }

    figtitles = 'pulseshaping' if learn_tx else 'rxfilt'
    if learn_tx and learn_rx:
        figtitles = 'both'

    figprefix = f"{FIGPREFIX}_{figtitles}_vpp{eam_config['pp_voltage']}_chanlength{smf_config['fiber_length']}"

    if not os.path.exists(FIGURE_DIR):
        os.mkdir(FIGURE_DIR)

    # Create modulation scheme
    modulation_scheme = komm.PAModulation(order=mod_order)
    print(f'Constellation: {modulation_scheme}')

    # Set up random seed and generate random bit sequence
    seed = 5
    random_obj = np.random.default_rng(seed=seed)

    # Optimization parameters
    learning_rate = 1e-3
    batch_size = 1000

    # Initialize learnable transmission system
    imdd_system = IntensityModulationChannel(sps=samples_per_symbol, baud_rate=baud_rate,
                                             learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                             learn_tx=learn_tx, learn_rx=learn_rx, rrc_rolloff=rrc_rolloff,
                                             tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr,
                                             adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                             tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                             smf_config=smf_config, photodiode_config=photodiode_config, eam_config=eam_config,
                                             ideal_modulator=ideal_modulator)

    imdd_system.initialize_optimizer()

    # Get the LPF filters
    adc_filter_b, adc_filter_a = None, None
    dac_filter_b, dac_filter_a = None, None
    if adc_bwl_relative_cutoff:
        adc_filter_b, adc_filter_a = imdd_system.adc.get_lpf_filter()
    if dac_bwl_relative_cutoff:
        dac_filter_b, dac_filter_a = imdd_system.dac.get_lpf_filter()

    # Generate training data
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_train)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)

    # Fit
    if learn_tx or learn_rx:
        imdd_system.optimize(a)

    # Generate validation data and caclulate SER on that with learned filters
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_val)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)
    rx_out = imdd_system.evaluate(a, decimate=False)
    ahat = rx_out[::samples_per_symbol]
    ser, delay = calc_ser_pam(ahat, a, discard=100)
    print(f"SER: {ser} (delay: {delay})")

    # Plot learned filters vs. matched
    filter_amp_min_db = -40.0
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12.5))
    for sys, label in zip([imdd_system],
                          ['E2E']):
        txfilt = sys.get_pulse_shaping_filter()
        rxfilt = sys.get_rx_filter()

        # Calculate the total response of the system (includuing LPFs)
        total_response = np.copy(txfilt)
        if dac_bwl_relative_cutoff:
            total_response = lfilter(dac_filter_b, dac_filter_a, total_response)
        if adc_bwl_relative_cutoff:
            total_response = lfilter(adc_filter_b, adc_filter_a, total_response)
        total_response = np.convolve(total_response, rxfilt)

        # First row - time domain
        ax[0, 0].plot(txfilt, '--', label=label)
        ax[0, 1].plot(rxfilt, '--', label=label)
        ax[0, 2].plot(total_response, '--', label=label)

        # Second row - frequency domain
        plot_fft_filter_response(txfilt, ax[1, 0], Ts=sys.Ts, plot_label=label)
        plot_fft_filter_response(rxfilt, ax[1, 1], Ts=sys.Ts, plot_label=label)
        plot_fft_filter_response(total_response, ax[1, 2], Ts=sys.Ts, plot_label=label)

    # Plot the ADC/DAC LPF filters on top of respective Tx and Rx
    if dac_bwl_relative_cutoff:
        plot_fft_ab_response(dac_filter_b, dac_filter_a, ax[1, 0], Ts=sys.Ts, plot_label='DAC')

    if adc_bwl_relative_cutoff:
        plot_fft_ab_response(adc_filter_b, adc_filter_a, ax[1, 1], Ts=sys.Ts, plot_label='ADC')

    # Pretty labeling
    ax[0, 0].set_ylabel('Time domain')
    ax[1, 0].set_ylabel('Fq domain')
    ax[0, 0].set_title('Pulse Shaping (learned)' if learn_tx else 'Pulse Shaping')
    ax[0, 1].set_title('Rx filter (learned)' if learn_rx else 'Rx filter')
    ax[0, 2].set_title('Total response (LPFs)')
    ax[1, 0].legend(loc='lower center')
    ax[1, 1].legend(loc='lower center')
    for i in range(3):
        __, ymax = ax[1, i].get_ylim()
        ax[1, i].set_ylim(filter_amp_min_db, ymax)

    plt.tight_layout()

    # Plot distribution of symbols
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(ahat, bins=100, density=True)

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_symbol_dist.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_symbol_dist.png"), dpi=DPI)

    # Calc theory SER
    esn0_db = imdd_system.get_esn0_db()
    ser_theory = calc_theory_ser_pam(mod_order, esn0_db)
    ser_mf_conf = 1.96 * np.sqrt((ser * (1 - ser) / (n_symbols_val)))
    print(f"Theoretical SER: {ser_theory} (EsN0: {esn0_db:.3f} [dB])")
    print(f"95pct confidence (+-) {ser_mf_conf}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_bar(['E2E', 'Theory'],
             [np.log10(x) for x in [ser, ser_theory]],
             ax)

    # Plot voltage-to-absorption function - compare with (Liang and Kahn)
    v = torch.linspace(-4.0, 0.0, 1000)
    fig, ax = plt.subplots(figsize=FIGSIZE, ncols=2)
    if ideal_modulator:
        ax[0].plot(v, v)
        xin = torch.linspace(-1.0, 1.0, 1000)
        ax[1].plot(xin, imdd_system.modulator.forward(xin))
        fig.suptitle('Ideal modulator...')
    else:
        alpha_db = imdd_system.modulator.spline_object.evaluate(v)
        ax[0].plot(v, alpha_db)
        ax[0].plot(imdd_system.modulator.x_knee_points, imdd_system.modulator.y_knee_points, 'ro')
        ax[0].axvline(imdd_system.modulator.voltage_min, color='k', linestyle='--')
        ax[0].axvline((imdd_system.modulator.voltage_min - imdd_system.modulator.pp_voltage), color='k', linestyle='--')
        ax[0].set_xlabel('Voltage')
        ax[0].set_ylabel('Absorption [dB]')
        ax[0].invert_xaxis()
        ax[0].invert_yaxis()
        ax[0].grid()

        xin = torch.linspace(-imdd_system.dac.dac_min_max, imdd_system.dac.dac_min_max, 1000)
        vin = (xin + (imdd_system.dac.dac_min_max)) / (2 * imdd_system.dac.dac_min_max)
        ax[1].plot(xin, imdd_system.modulator.forward(xin))
        ax[1].set_xlabel('Digital signal')
        ax[1].set_ylabel('Optical field amplitude')
        ax[1].grid()

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_modulator_response.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_modulator_response.png"), dpi=DPI)

    # Plot channel phase response
    f, channelH = imdd_system.channel.get_fq_filter(len(a) * samples_per_symbol)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(f, np.unwrap(np.angle(channelH)))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Phase response')
    ax.set_title(f"Phase response of channel. Fiber length: {smf_config['fiber_length']} [km]")
    ax.grid()

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_channel_phase.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_channel_phase.png"), dpi=DPI)

    # Eyediagram
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_eyediagram(rx_out, ax, imdd_system.Ts, samples_per_symbol, decimation=n_symbols_val//int(1e4))
    ax.set_title('Eyediagram')
    ax.set_xlabel('time [s]')

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_eyediagram.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_eyediagram.png"), dpi=DPI)

    plt.show()
