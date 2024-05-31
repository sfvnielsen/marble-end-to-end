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

from lib.utility import calc_ser_pam
from lib.systems import PulseShapingIMwithWDM, RxFilteringIMwithWDM, JointTxRxIMwithWDM, LinearFFEIMwithWDM
from lib.plotting import plot_fft_filter_response

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 14}

text = {'usetex': True}

mpl.rc('font', **font)
mpl.rc('text', **text)

FIGSIZE = (12.5, 7.5)
DPI = 150
FIGURE_DIR = 'figures'
FIGPREFIX = 'e2e_imdd_wdm'


if __name__ == "__main__":
    # Define simulation parameters
    save_figures = True
    n_symbols_train = int(1e6)
    n_symbols_val = int(1e6)  # number of symbols used for SER calculation
    samples_per_symbol = 8
    baud_rate = int(100e9)
    mod_order = 4  # PAM
    rrc_rolloff = 0.01
    learn_tx, tx_filter_length = True, 51
    learn_rx, rx_filter_length = True, 51
    dac_bwl_relative_cutoff = 0.9  # low-pass filter cuttoff relative to information bandwidth
    adc_bwl_relative_cutoff = 0.9
    adc_bitres = 5
    dac_bitres = 5
    use_1clr = True
    dac_voltage_pp = 3.0
    dac_voltage_bias = -1.5

    # Configuration of electro absorption modulator
    modulator_type = 'eam'
    modulator_config = {
        'laser_power_dbm': -6.0,
        'linewidth_enhancement': 2.0,
        'linear_absorption': False
    }

    # Channel configuration - single model fiber
    smf_config = {
        'fiber_length': 0.5,
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

    # WDM config - used during evaluation
    wdm_channel_spacings = np.array([baud_rate * 1.05, baud_rate * 1.25, baud_rate * 1.5, baud_rate * 2.0, baud_rate * 3.0])
    wdm_n_channels = 3

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

    # Initialize learnable transmission system(s)
    joint_tx_rx = JointTxRxIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                     learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                     rrc_rolloff=rrc_rolloff,
                                     dac_voltage_bias=dac_voltage_bias, dac_voltage_pp=dac_voltage_pp,
                                     tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr,
                                     adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                     tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                     smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                     modulator_type=modulator_type)

    ps_sys = PulseShapingIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                   learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                   rrc_rolloff=rrc_rolloff,
                                   dac_voltage_bias=dac_voltage_bias, dac_voltage_pp=dac_voltage_pp,
                                   tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr,
                                   adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                   tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                   smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                   modulator_type=modulator_type)

    rxf_sys = RxFilteringIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                   learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                   rrc_rolloff=rrc_rolloff,
                                   dac_voltage_bias=dac_voltage_bias, dac_voltage_pp=dac_voltage_pp,
                                   tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr,
                                   adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                   tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                   smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                   modulator_type=modulator_type)
    
    ffe_sys = LinearFFEIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                 learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                 rrc_rolloff=rrc_rolloff, ffe_n_taps=35,
                                 dac_voltage_bias=dac_voltage_bias, dac_voltage_pp=dac_voltage_pp,
                                 tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr,
                                 adc_bwl_relative_cutoff=adc_bwl_relative_cutoff, dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                                 tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                 smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                 modulator_type=modulator_type)
    

    # Generate training data
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_train)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a_train = modulation_scheme.modulate(bit_sequence)

    # Generate validation data and caclulate SER on that with learned filters for all WDM configurations
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_val)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)

    res_dict = dict()
    for imdd_system, system_label in zip([joint_tx_rx, ps_sys, rxf_sys, ffe_sys],
                                         ['PS \& RxF', 'PS', 'RxF', 'RRC \& FFE']):

        imdd_system.initialize_optimizer()

        # Fit
        imdd_system.optimize(a_train)

        ser_pr_spacing = np.zeros_like(wdm_channel_spacings)
        for cs, channel_spacing in enumerate(wdm_channel_spacings):
            rx_out = imdd_system.evaluate(a, channel_spacing_hz=channel_spacing, n_channels=wdm_n_channels)
            ser, delay = calc_ser_pam(rx_out, a, discard=100)
            print(f"SER: {ser} (delay: {delay}) [channel spacing: {channel_spacing / 1e9} GHz]")

            ser_pr_spacing[cs] = ser
        
        res_dict[system_label] = ser_pr_spacing
        

    # Plot SER as a function of channel spacing
    fig, ax = plt.subplots()
    for meth, sercurve in res_dict.items():
        ax.plot(wdm_channel_spacings / 1e9, sercurve, label=meth, marker='o')
    ax.set_yscale('log')
    ax.set_xlabel('Channel spacing [GHz]')
    ax.set_ylabel('SER')
    ax.grid()
    ax.legend()
    fig.tight_layout()

    # Plot example of WDM signal
    fig, ax = plt.subplots()
    symbols_up = torch.zeros((len(a) * samples_per_symbol, ), dtype=torch.float64)
    symbols_up[::samples_per_symbol] = torch.from_numpy(a)
    with torch.no_grad():
        tx_wdm = joint_tx_rx.eval_tx(symbols_up, n_channels=wdm_n_channels, batch_size=int(1e5),
                                     channel_spacing_hz=wdm_channel_spacings[len(wdm_channel_spacings) // 2])
        tx_chan = joint_tx_rx.eval_channel_selection(tx_wdm, wdm_channel_spacings[len(wdm_channel_spacings) // 2])
    ax.psd(tx_wdm, Fs=1 / joint_tx_rx.Ts, label='Tx WDM', sides='twosided')
    ax.psd(tx_chan, Fs=1 / joint_tx_rx.Ts, label='Tx Chan select', sides='twosided')

    ax.set_title(f"Example: Channel spacing {wdm_channel_spacings[len(wdm_channel_spacings) // 2] / 1e9} GHz")
    fig.tight_layout()

    # Plot the learned filters
    fig, ax = plt.subplots(nrows=2)
    filter_amp_min_db = -40.0
    for imdd_system, system_label in zip([joint_tx_rx, ps_sys, rxf_sys, ffe_sys],
                                         ['PS \& RxF', 'PS', 'RxF', 'RRC \& FFE']):

        plot_fft_filter_response(imdd_system.get_pulse_shaping_filter(), ax[0], Ts=joint_tx_rx.Ts, plot_label=system_label)
        plot_fft_filter_response(imdd_system.get_rx_filter(), ax[1], Ts=joint_tx_rx.Ts, plot_label=system_label)
    
    ax[0].set_title('Tx filter')
    ax[1].set_title('Rx filter')

    ax[0].legend()
    leg_lines, leg_labels = ax[0].get_legend_handles_labels()
    ax[0].get_legend().remove()
    fig.legend(leg_lines, leg_labels, loc='upper center', ncol=len(leg_labels))

    for this_ax in ax:
        this_ax.set_ylim(*(filter_amp_min_db, None))

    plt.show()
