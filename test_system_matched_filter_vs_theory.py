"""
    Test that matched filter and theory matches for the AWGN channel
"""

import numpy as np
import komm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from commpy.filters import rrcosfilter

from lib.systems import MatchedFilterAWGN
from lib.utility import calc_theory_ser_pam, calc_ser_pam

if __name__ == "__main__":
    # Parameters to be used
    order = 4  # modulation order
    n_symbols = int(1e6)

    normalize_after_tx = True
    snr_db = np.arange(-6.0, 8.0, 2.0)

    sym_rate = int(10e6)  # baud - number of transmitted symbols pr second
    sym_length = 1 / sym_rate
    samples_pr_symbol = 16
    Ts = sym_length / samples_pr_symbol  # effective sampling interval
    reps = 3

    # Pulse shaping - used root raised cosine filter
    pulse_length_in_symbols = 64
    rolloff = 0.1

    # Create modulation scheme
    modulation_scheme = komm.PAModulation(order=order)

    print(modulation_scheme)

    # Set up random seed and generate random bit sequence
    seed = 12346
    random_obj = np.random.default_rng(seed=seed)

    # Create empty list for concatenating result dicts (later converted to Pandas DataFrame)
    results = []

    # Create system object
    mf_system = MatchedFilterAWGN(sps=samples_pr_symbol, baud_rate=sym_rate, constellation=modulation_scheme.constellation,
                                  normalize_after_tx=True, rrc_length_in_symbols=pulse_length_in_symbols, rrc_rolloff=rolloff, snr_db=-90)

    esn0_db_list = []

    # Loop over experiments
    for snr in snr_db:
        print(f"Running experiments for SNR: {snr}", flush=True)
        mf_system.set_snr(snr)
        avg_ser = 0.0

        for nrep in range(reps):
            # Initialize result dictionary for this experiment
            res_dict = dict()
            res_dict['snr'] = snr
            res_dict['rep'] = nrep
            res_dict['rrc_rolloff'] = rolloff
            res_dict['method'] = 'Matched filter'
            res_dict['esn0_db'] = mf_system.get_esn0_db()

            # Create data - matched filter already applied and then decimated
            syms = random_obj.choice(modulation_scheme.constellation, size=(n_symbols,), replace=True)
            rx = mf_system.evaluate(syms)
            ser, __ = calc_ser_pam(rx, syms)
            res_dict['ser'] = ser

            results.append(res_dict)
            avg_ser += ser / reps

        print(f"Average SER: {avg_ser} at EsN0: {mf_system.get_esn0_db()}")
        esn0_db_list.append(mf_system.get_esn0_db())

    # Calculate theoretical SER
    for esn0_db in esn0_db_list:
        theo_ser = calc_theory_ser_pam(order, esn0_db)

        res_dict = dict()
        res_dict['snr'] = np.nan
        res_dict['rep'] = np.nan
        res_dict['rrc_rolloff'] = np.nan
        res_dict['ser'] = theo_ser
        res_dict['esn0_db'] = esn0_db
        res_dict['method'] = 'Theoretical'
        results.append(res_dict)
        print(f"Method: Theoretical, SER: {theo_ser}", flush=True)


    # Gather resutls as DataFrame and plot using Seaborn
    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(ncols=2)
    t, g = rrcosfilter(pulse_length_in_symbols * samples_pr_symbol, rolloff, sym_length, 1 / Ts)
    g = g / np.linalg.norm(g)
    ax[0].plot(t, g, label=f'rolloff = {rolloff}')
    ax[0].grid()
    ax[0].legend()

    lplot = sns.lineplot(results_df, x='esn0_db', y='ser', hue='method', ax=ax[1])
    lplot.set(yscale='log')
    ax[1].grid()

    plt.show()
