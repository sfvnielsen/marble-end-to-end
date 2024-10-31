"""
    Script that runs end-to-end learning, calculates SER and plots system response
"""

import os
import komm
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    figprefix = f"{FIGPREFIX}"
    n_symbols_train = int(1e6)
    n_symbols_val = int(1e6)  # number of symbols used for SER calculation
    samples_per_symbol = 8 
    baud_rate = int(100e9)
    mod_order = 4  # PAM
    rrc_rolloff = 0.01
    tx_filter_length = 65
    rx_filter_length = 65
    eval_adc_bitres = 5
    eval_dac_bitres = 5
    lr_schedule = 'oneclr'

    # Configuration for DAC (and ADC)
    dac_config = {
        'peak_to_peak_voltage': 2.0,
        'bias_voltage': 'positive_vpp',
        'bwl_cutoff': 45e9,  # Hz
        'learnable_normalization': True,
        'learnable_bias': False,
        'filter_type': 'bessel',
        'lpf_order': 5
    }

    adc_bwl_cutoff_hz = 45e9  # same as dac

    # Configuration of electro absorption modulator
    modulator_type = 'ideal'
    modulator_config = {
        'laser_power_dbm': -11.0,
    }

    # Channel configuration - single model fiber
    fiber_type = 'ssfm'
    fiber_config = {
        'fiber_length': 1.0,
        'attenuation': 0.0,
        'carrier_wavelength': 1270,
        'zero_dispersion_wavelength': 1310,
        'dispersion_slope': 0.092,
        'gamma': 1.3,
        'step_length': 0.25
    }

    # Configuration of the photodiode
    photodiode_config = {
        'responsivity': 1.0,
        'temperature': 20.0,
        'dark_current': 10e-9,
        'impedance_load': 50.0
    }

    # WDM config
    wdm_channel_spacing = baud_rate * 1.5
    wdm_channel_selection_rel_cutoff = 1.1

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
                                     wdm_channel_spacing_hz=wdm_channel_spacing,
                                     wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                                     tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, lr_schedule=lr_schedule,
                                     tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                     fiber_config=fiber_config, fiber_type=fiber_type, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                     modulator_type=modulator_type, dac_config=dac_config, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz)

    ps_sys = PulseShapingIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                   learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                   rrc_rolloff=rrc_rolloff,
                                   wdm_channel_spacing_hz=wdm_channel_spacing,
                                   wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                                   tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, lr_schedule=lr_schedule,
                                   tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                   fiber_config=fiber_config, fiber_type=fiber_type, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                   modulator_type=modulator_type, dac_config=dac_config, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz)

    rxf_sys = RxFilteringIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                   learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                   rrc_rolloff=rrc_rolloff,
                                   wdm_channel_spacing_hz=wdm_channel_spacing,
                                   wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                                   tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, lr_schedule=lr_schedule,
                                   tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                   fiber_config=fiber_config, fiber_type=fiber_type, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                   modulator_type=modulator_type, dac_config=dac_config, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz)
    
    ffe_sys = LinearFFEIMwithWDM(sps=samples_per_symbol, baud_rate=baud_rate,
                                 learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                 rrc_rolloff=rrc_rolloff, ffe_n_taps=rx_filter_length,
                                 wdm_channel_spacing_hz=wdm_channel_spacing,
                                 wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                                 tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, lr_schedule=lr_schedule,
                                 tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                                 fiber_config=fiber_config, fiber_type=fiber_type, photodiode_config=photodiode_config, modulator_config=modulator_config,
                                 modulator_type=modulator_type, dac_config=dac_config, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz)
    

    # Generate training data
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_train)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a_train = modulation_scheme.modulate(bit_sequence)

    # Generate validation data and caclulate SER on that with learned filters for all WDM configurations
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_val)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)

    res_dicts = []
    for imdd_system, system_label in zip([joint_tx_rx, ps_sys, rxf_sys, ffe_sys],
                                         ['PS \& RxF', 'PS', 'RxF', 'RRC \& FFE']):

        imdd_system.initialize_optimizer()

        # Fit
        imdd_system.optimize(a_train)

        # Evaluate
        rx_out = imdd_system.evaluate(a, dac_bitres=eval_dac_bitres, adc_bitres=eval_adc_bitres)
        ser, delay = calc_ser_pam(rx_out, a, discard=100)
        print(f"SER: {ser} (delay: {delay}) [channel spacing: {wdm_channel_spacing / 1e9} GHz]")

        # Save results - prep for DataFrame
        res_dict = dict()
        res_dict['method'] = system_label
        res_dict['ser'] = ser
        res_dicts.append(res_dict)
        

    # Plot SER as a function of channel spacing
    fig, ax = plt.subplots()
    sns.barplot(data=pd.DataFrame(res_dicts), x='method', y='ser', ax=ax)
    ax.set_yscale('log')
    ax.set_ylabel('SER')
    ax.grid()
    fig.tight_layout()

    # Plot example of WDM signal
    systems_under_test = [joint_tx_rx, ps_sys, rxf_sys, ffe_sys]
    fig, ax = plt.subplots(ncols=len(systems_under_test), figsize=(12, 5))
    symbols_up = torch.zeros((len(a) * samples_per_symbol, ), dtype=torch.float64)
    symbols_up[::samples_per_symbol] = torch.from_numpy(a)

    for s, (sys, label) in enumerate(zip([joint_tx_rx, ps_sys, rxf_sys, ffe_sys],
                                         ['PS \& RxF', 'PS', 'RxF', 'RRC \& FFE'])):
        with torch.no_grad():
            tx_wdm = sys.eval_tx(symbols_up, channel_spacing_hz=wdm_channel_spacing, batch_size=int(1e5))
            tx_chan = sys.channel_selection_filter.forward(tx_wdm)
        ax[s].psd(tx_wdm, Fs=1 / sys.Ts, label='Tx WDM', sides='twosided')
        ax[s].psd(tx_chan, Fs=1 / sys.Ts, label='Tx Chan select', sides='twosided')
        ax[s].set_title(label)

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
    
    fig.tight_layout()

    if save_figures:
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_filters.eps"), format='eps')
        fig.savefig(os.path.join(FIGURE_DIR, f"{figprefix}_filters.png"), dpi=DPI)

    plt.show()
