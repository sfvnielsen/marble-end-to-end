"""
    Script that runs end-to-end learning, calculates SER and plots system response
"""

import os
import komm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from lib.utility import calc_ser_pam, calc_theory_ser_pam
from lib.systems import NonLinearISIChannel
from lib.plotting import plot_bar, plot_fft_filter_response, plot_fft_ab_response

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 20}

text = {'usetex': True}

mpl.rc('font', **font)
mpl.rc('text', **text)

FIGSIZE = (12.5, 7.5)
DPI = 150
FIGURE_DIR = 'figures'
FIGPREFIX = 'e2e_non_lin'

if __name__ == "__main__":
    # Define simulation parameters
    save_figures = False
    n_symbols_train = int(15e5)
    n_symbols_val = int(5e5)  # number of symbols used for SER calculation
    samples_per_symbol = 4
    baud_rate = int(100e9)
    train_snr_db = 8.0  # SNR (EsN0) at which the system in trained in
    eval_snr_db = train_snr_db
    mod_order = 4  # PAM
    rrc_rolloff = 0.5
    learn_tx, tx_filter_length = True, 35
    learn_rx, rx_filter_length = True, 35
    dac_bwl_relative_cutoff = 0.9  # low-pass filter cuttoff relative to bandwidth of the RRC pulse
    adc_bwl_relative_cutoff = 0.9
    use_brickwall = False  # use brickwall filter instead of Bessel in the ADC/DAC (Experimental)
    use_1clr = True  # learning rate scheduling of the optimizer

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
    non_linear_coefs = (0.9, 0.2, -0.1)
    non_lin_system = NonLinearISIChannel(non_linear_coefficients=non_linear_coefs,
                                         isi_filter1=np.array([0.2, 0.9, 0.01, 0.02, 0.03]),
                                         isi_filter2=np.array([0.2, 0.001, 0.9,-0.01]),
                                         sps=samples_per_symbol, esn0_db=train_snr_db, baud_rate=baud_rate,
                                         learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                         learn_tx=learn_tx, learn_rx=learn_rx, rrc_rolloff=rrc_rolloff,
                                         tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr, use_brickwall=use_brickwall,
                                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                         tx_filter_init_type='rrc', rx_filter_init_type='rrc')

    non_lin_system.initialize_optimizer()

    # Get the LPF filters
    adc_filter_b, adc_filter_a = None, None
    dac_filter_b, dac_filter_a = None, None
    if adc_bwl_relative_cutoff:
        adc_filter_b, adc_filter_a = non_lin_system.adc.get_lpf_filter()
    if dac_bwl_relative_cutoff:
        dac_filter_b, dac_filter_a = non_lin_system.dac.get_lpf_filter()

    # Generate training data
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_train)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)

    # Fit
    if learn_tx or learn_rx:
        non_lin_system.optimize(a)

    # Generate validation data and caclulate SER on that with learned filters
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_val)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)
    non_lin_system.set_esn0_db(eval_snr_db)
    ahat = non_lin_system.evaluate(a)
    ser, delay = calc_ser_pam(ahat, a, discard=100)
    print(f"SER: {ser} (delay: {delay})")


    # Plot learned filters vs. matched
    # FIXME: How to plot this with non-linearity in channel?


    # Plot learned filters vs. matched
    filter_amp_min_db = -80.0
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12.5))
    for sys, label in zip([non_lin_system],
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
        total_response = np.convolve(total_response, non_lin_system.get_total_isi())

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
    ax[0, 2].set_title('Total response (LPFs + ISI)')
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
    esn0_db = non_lin_system.get_esn0_db()
    ser_theory = calc_theory_ser_pam(mod_order, esn0_db)
    ser_mf_conf = 1.96 * np.sqrt((ser * (1 - ser) / (n_symbols_val)))
    print(f"Theoretical SER: {ser_theory} (EsN0: {esn0_db:.3f} [dB])")
    print(f"95pct confidence (+-) {ser_mf_conf}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_bar(['E2E', 'Theory'],
             [np.log10(x) for x in [ser, ser_theory]],
             ax)
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.linspace(-1.5, 1.5, 1000)
    ax.plot(x, non_lin_system.non_linear_function(x))
    ax.set_title('Non-linear function')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_non_linear_fun.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_non_linear_fun.png"), dpi=DPI)


    plt.show()
