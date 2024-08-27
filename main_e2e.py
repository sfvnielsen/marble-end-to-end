"""
    Script that runs end-to-end learning, calculates SER and plots system response
"""

import os
import komm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
from scipy.fft import fftshift

from lib.utility import calc_ser_pam, calc_theory_ser_pam
from lib.systems import BasicAWGNwithBWL, LinearFFEAWGNwithBWL
from lib.plotting import plot_bar, plot_fft_filter_response, plot_fft_ab_response, plot_pole_zero, plot_fft

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 20}

text = {'usetex': True}

mpl.rc('font', **font)
mpl.rc('text', **text)

FIGSIZE = (8.0, 4.0)
DPI = 150
FIGURE_DIR = 'figures'
FIGPREFIX = 'e2e'

if __name__ == "__main__":
    # Define simulation parameters
    save_figures = False
    n_symbols_train = int(15e5)
    n_symbols_val = int(5e5)  # number of symbols used for SER calculation
    samples_per_symbol = 8
    baud_rate = int(100e9)
    train_snr_db = 12.0  # SNR (EsN0) at which the training is done
    eval_snr_db = 8.0
    mod_order = 4  # PAM
    rrc_rolloff = 0.01  # for initialization
    learn_tx, tx_filter_length = False, 15
    learn_rx, rx_filter_length = True, 15
    dac_bwl_relative_cutoff = 0.9  # low-pass filter cuttoff relative to bandwidth of the baseband signal
    adc_bwl_relative_cutoff = 0.9
    use_1clr = True  # learning rate scheduling of the optimizer
    use_brickwall = False

    figtitles = 'pulseshaping' if learn_tx else 'rxfilt'
    if learn_tx and learn_rx:
        figtitles = 'both'

    figprefix = f"{FIGPREFIX}_{figtitles}_adc{adc_bwl_relative_cutoff}_dac{dac_bwl_relative_cutoff}"

    if not os.path.exists(FIGURE_DIR):
        os.mkdir(FIGURE_DIR)

    # Create modulation scheme
    modulation_scheme = komm.PAModulation(order=mod_order)
    print(f'Constellation: {modulation_scheme}')

    # Set up random seed and generate random bit sequence
    seed = 5
    random_obj = np.random.default_rng(seed=seed)

    # Optimization parameters
    learning_rate = 1e-4
    batch_size = 1000

    # Initialize learnable transmission system
    awgn_system = BasicAWGNwithBWL(sps=samples_per_symbol, esn0_db=train_snr_db, baud_rate=baud_rate,
                                   learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                   learn_tx=learn_tx, learn_rx=learn_rx, rrc_rolloff=rrc_rolloff,
                                   tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr, use_brickwall=use_brickwall,
                                   adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                   tx_filter_init_type='rrc', rx_filter_init_type='rrc')

    awgn_system.initialize_optimizer()

    # Get the LPF filters
    adc_filter_b, adc_filter_a = None, None
    dac_filter_b, dac_filter_a = None, None
    if adc_bwl_relative_cutoff:
        adc_filter_b, adc_filter_a = awgn_system.adc.get_filters()
    if dac_bwl_relative_cutoff:
        dac_filter_b, dac_filter_a = awgn_system.dac.get_filters()

    # Generate training data
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_train)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a_train = modulation_scheme.modulate(bit_sequence)

    # Fit
    if learn_tx or learn_rx:
        awgn_system.optimize(a_train)

    # Generate validation data and caclulate SER on that with learned filters
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_val)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a_val = modulation_scheme.modulate(bit_sequence)
    awgn_system.set_esn0_db(eval_snr_db)
    y_rx = awgn_system.evaluate(a_val, decimate=False)
    ahat = y_rx[0::samples_per_symbol]
    ser, delay = calc_ser_pam(ahat, a_val, discard=100)
    print(f"SER: {ser} (delay: {delay})")

    # Plot signal PSD after Rx filter
    fig, ax = plt.subplots()
    plot_fft(y_rx, ax=ax, Ts=awgn_system.Ts)
    ymin, ymax = ax.get_ylim()
    ax.vlines([-baud_rate/2, baud_rate/2], ymin, ymax, 'k')
    ax.set_title('PSD of signal after matched filter')

    # Compare to standard non-optimized RRC + equalizer
    rrc_pulse_length_in_syms = tx_filter_length // samples_per_symbol + 1
    awgn_ffe_system = LinearFFEAWGNwithBWL(sps=samples_per_symbol, esn0_db=train_snr_db, baud_rate=baud_rate,
                                           learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                           ffe_n_taps=rx_filter_length, rrc_rolloff=rrc_rolloff,
                                           tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr, use_brickwall=use_brickwall,
                                           adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                           tx_filter_init_type='rrc', rx_filter_init_type='rrc')
    
    awgn_ffe_system.initialize_optimizer()
    awgn_ffe_system.optimize(a_train)
    awgn_ffe_system.set_esn0_db(eval_snr_db)
    ahat_ffe = awgn_ffe_system.evaluate(a_val)
    ser_ffe, delay_ffe = calc_ser_pam(ahat_ffe, a_val, discard=100)
    print(f"SER (FFE): {ser_ffe} (delay: {delay_ffe})")

    # Plot learned filters vs. matched
    filter_amp_min_db = -80.0
    n_fft = 512 * samples_per_symbol
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12.5, 7.5))
    for sys, label in zip([awgn_system, awgn_ffe_system],
                   ['E2E', 'RRC+FFE']):
        txfilt = sys.get_pulse_shaping_filter()
        rxfilt = sys.get_rx_filter()

        # Calculate the total response of the system (includuing LPFs)
        total_response = np.copy(txfilt)
        f, total_response_fq = freqz(txfilt, 1, worN=n_fft, whole=True, fs=1/sys.Ts)
        if dac_bwl_relative_cutoff:
            total_response = lfilter(dac_filter_b, dac_filter_a, total_response)
            _, Hfq = freqz(dac_filter_b, dac_filter_a, worN=n_fft, whole=True, fs=1/sys.Ts)
            total_response_fq  = total_response_fq * Hfq
        if adc_bwl_relative_cutoff:
            total_response = lfilter(adc_filter_b, adc_filter_a, total_response)
            _, Hfq = freqz(dac_filter_b, dac_filter_a, worN=n_fft, whole=True, fs=1/sys.Ts)
            total_response_fq  = total_response_fq * Hfq
        total_response = np.convolve(total_response, rxfilt)
        _, Hfq = freqz(rxfilt, 1, worN=n_fft, whole=True, fs=1/sys.Ts)
        total_response_fq = total_response_fq * Hfq

        if 'FFE' in label:
            heq = sys.equaliser.filter.get_filter()
            total_response = np.convolve(total_response, heq)
            _, Hfq = freqz(heq, 1, worN=n_fft, whole=True, fs=1/sys.Ts)
            total_response_fq *= Hfq

        # First row - time domain
        ax[0, 0].plot(txfilt, '--', label=label)
        ax[0, 1].plot(rxfilt, '--', label=label)
        ax[0, 2].plot(total_response, '--', label=label)

        # Second row - frequency domain
        plot_fft_filter_response(txfilt, ax[1, 0], Ts=sys.Ts, plot_label=label)
        plot_fft_filter_response(rxfilt, ax[1, 1], Ts=sys.Ts, plot_label=label)
        fqs_freqz = np.arange(-len(f)//2, len(f)//2) / (len(f) * sys.Ts)
        ax[1, 2].plot(fqs_freqz, 20.0 * np.log10(np.absolute(fftshift(total_response_fq))), label=label)

    # Plot the ADC/DAC LPF filters on top of respective Tx and Rx
    if dac_bwl_relative_cutoff:
        plot_fft_ab_response(dac_filter_b, dac_filter_a, ax[1, 0], Ts=sys.Ts, plot_label='DAC')

    if adc_bwl_relative_cutoff:
        plot_fft_ab_response(adc_filter_b, adc_filter_a, ax[1, 1], Ts=sys.Ts, plot_label='ADC')

    # Pretty labeling
    ax[0, 0].set_ylabel('Time domain')
    ax[1, 0].set_ylabel('Fq domain')
    ax[0, 0].set_title('Pulse-shaper')
    ax[0, 1].set_title('Receiver filter')
    ax[0, 2].set_title('Total response (including LPF)')
    ax[1, 0].legend(loc='lower center')
    ax[1, 1].legend(loc='lower center')
    for i in range(3):
        ax[1, i].set_ylim(ymin=filter_amp_min_db)

    ax[1, 2].grid(True)
    plt.tight_layout()
    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_system_response.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_system_response.png"), dpi=DPI)


    # Plot the pole-zero plot of the system response up until before the Tx filter
    fig, axs = plt.subplots(figsize=FIGSIZE, ncols=3)
    tx_filter = awgn_system.get_pulse_shaping_filter()
    total_response_b = np.copy(tx_filter)
    total_response_a = [1]
    if dac_bwl_relative_cutoff:
        total_response_b = np.polymul(dac_filter_b, total_response_b)
        total_response_a = np.polymul(dac_filter_a, total_response_a)
    if adc_bwl_relative_cutoff:
        total_response_b = np.polymul(adc_filter_b, total_response_b)
        total_response_a = np.polymul(adc_filter_a, total_response_a)
    
    plot_pole_zero((tx_filter, 1), axs[0])
    axs[0].set_title(f'Tx filter (len={len(tx_filter)})')
    axs[0].set_ylabel('Imag(z)')

    plot_pole_zero((dac_filter_b, dac_filter_a), axs[1])
    axs[1].set_title(f'Bessel filter (order={len(dac_filter_b)})')

    plot_pole_zero((total_response_b, total_response_a), axs[2])
    axs[2].set_title('System response (without Rx)')

    for ax in axs:
        ax.set_xlabel('Real(z)')

    fig.suptitle('Pole-zero plots')
    plt.tight_layout()

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_system_response_pole_zero.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_system_response_pole_zero.png"), dpi=DPI)

    # Plot distribution of symbols
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(ahat, bins=100, density=True)

    # Calc theory SER
    esn0_db = awgn_system.get_esn0_db()
    ser_theory = calc_theory_ser_pam(mod_order, esn0_db)
    ser_mf_conf = 1.96 * np.sqrt((ser_ffe * (1 - ser_ffe) / (n_symbols_val)))
    print(f"Theoretical SER: {ser_theory} (EsN0: {esn0_db:.3f} [dB])")
    print(f"95pct confidence (+-) {ser_mf_conf}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_bar(['E2E', 'Matched filter', 'Theory'],
             [np.log10(x) for x in [ser, ser_ffe, ser_theory]],
             ax)

    plt.show()
