"""
    Vanilla learnable clock recovery in AWGN channel

    Learn one side (set other to RRC)
    Inside the channel - set a "sample_delay" - by which the signal will be rolled
    Let optimizer run and verify that we can find the correct sampling
    
    Compare to matched filtering with max_var_sampling estimation
"""

import komm
import numpy as np
import matplotlib.pyplot as plt

from lib.systems import BasicAWGNwithDelay, MatchedFilterAWGNwithDelay
from lib.utility import calc_ser_pam, calc_theory_ser_pam


if __name__ == "__main__":
    sample_delay = 2
    n_symbols_train = int(25e5)
    n_symbols_val = int(5e5)  # number of symbols used for SER calculation
    samples_per_symbol = 4
    baud_rate = int(100e6)
    train_snr_db = 16.0  # SNR (EsN0) at which the training is done
    eval_snr_db = 4.0
    mod_order = 4  # PAM
    rrc_rolloff = 0.5  # for initialization
    learn_tx, tx_filter_length = True, 20
    learn_rx, rx_filter_length = False, 20
    use_1clr = True  # learning rate scheduling of the optimizer

    # Create modulation scheme
    modulation_scheme = komm.PAModulation(order=mod_order)
    print(f'Constellation: {modulation_scheme}')

    # Set up random seed and generate random bit sequence
    seed = 5
    random_obj = np.random.default_rng(seed=seed)

    # Optimization parameters
    learning_rate = 5e-3
    batch_size = 1000

    # Initialize learnable transmission system
    awgn_system = BasicAWGNwithDelay(sps=samples_per_symbol, esn0_db=train_snr_db, baud_rate=baud_rate, sample_delay=sample_delay,
                                     learning_rate=learning_rate, batch_size=batch_size, constellation=modulation_scheme.constellation,
                                     learn_tx=learn_tx, learn_rx=learn_rx, rrc_rolloff=rrc_rolloff,
                                     tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length, use_1clr=use_1clr,
                                     tx_filter_init_type='rrc', rx_filter_init_type='rrc')

    awgn_system.initialize_optimizer()


    # Generate training data
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_train)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)

    # Fit
    if learn_tx or learn_rx:
        awgn_system.optimize(a)

    # Generate validation data and caclulate SER on that with learned filters
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols_val)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)
    awgn_system.set_esn0_db(eval_snr_db)
    ahat = awgn_system.evaluate(a)
    ser, delay = calc_ser_pam(ahat, a, discard=100)
    print(f"SER: {ser} (delay: {delay})")

    # Compare to standard non-optimized matched filtering
    awgn_mf_system = MatchedFilterAWGNwithDelay(sps=samples_per_symbol, esn0_db=eval_snr_db, baud_rate=baud_rate,
                                                constellation=modulation_scheme.constellation, sample_delay=sample_delay,
                                                rrc_length_in_symbols=tx_filter_length//samples_per_symbol, rrc_rolloff=rrc_rolloff)
    ahat_mf = awgn_mf_system.evaluate(a)
    ser_mf, delay_mf = calc_ser_pam(ahat_mf, a, discard=100)
    print(f"SER (Matched filter): {ser_mf} (delay: {delay_mf})")

    # Plot learned filters vs. matched
    filter_amp_min_db = -40.0
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 12.5))
    for sys, label in zip([awgn_system, awgn_mf_system],
                   ['E2E', 'Matched']):
        txfilt = sys.get_pulse_shaping_filter()
        rxfilt = sys.get_rx_filter()

        # Calculate the total response of the system (includuing LPFs)
        total_response = np.copy(txfilt)
        total_response = np.convolve(total_response, rxfilt)

        # First row - time domain
        ax[0].plot(txfilt, '--', label=label)
        ax[1].plot(rxfilt, '--', label=label)
        ax[2].plot(total_response, '--', label=label)

    # Pretty labeling
    ax[0].set_ylabel('Time domain')
    ax[0].set_title('Pulse Shaping (learned)' if learn_tx else 'Pulse Shaping')
    ax[1].set_title('Rx filter (learned)' if learn_rx else 'Rx filter')
    ax[2].set_title('Total response')
    ax[0].legend(loc='lower center')
    for i in range(3):
        ax[i].grid()

    # Plot distribution of symbols
    fig, ax = plt.subplots()
    ax.hist(ahat, bins=100, density=True)

    # Calc theory SER
    esn0_db = awgn_system.get_esn0_db()
    ser_theory = calc_theory_ser_pam(mod_order, esn0_db)
    ser_mf_conf = 1.96 * np.sqrt((ser_mf * (1 - ser_mf) / (n_symbols_val)))
    print(f"Theoretical SER: {ser_theory} (EsN0: {esn0_db:.3f} [dB])")
    print(f"95pct confidence (+-) {ser_mf_conf}")

    print(f"Within {20.0 * np.log10(ser/ser_theory)} [dB] of theory")

    plt.show()
