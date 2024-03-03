import numpy as np
from scipy.stats import norm
from scipy.signal import correlate


def symbol_sync(rx, tx, sps):
    """ Synchronizes tx symbols to the received signal, assumed to be oversample at sps
        Assumes that rx has been through a sampling recovery mechanism first

        Heavily inspired by the DSP library in https://github.com/edsonportosilva/OptiCommPy
    """
    rx_syms = rx[0::sps]
    delay = find_delay(tx, rx_syms)
    return np.roll(tx, -int(delay))


def find_delay(x, y):
    """ Find delay that maximizes correlation between real-parts

        Heavily inspired by the DSP library in https://github.com/edsonportosilva/OptiCommPy
    """
    return np.argmax(correlate(np.real(x), np.real(y))) - x.shape[0] + 1


def find_max_variance_sample(y, sps):
    """ Find sampling recovery compensation shift using the maximum variance method

        Heavily inspired by the DSP library in https://github.com/edsonportosilva/OptiCommPy
    """
    nsyms = len(y) // sps  # truncate to an integer num symbols
    yr = np.reshape(y[0:nsyms*sps], (-1, sps))
    var = np.var(yr, axis=0)
    max_variance_sample = np.argmax(var)
    return max_variance_sample


def calc_ser_pam(y_eq, a, discard=10):
    assert len(y_eq) == len(a)
    opt_delay = find_delay(y_eq, a)
    const = np.unique(a)
    ahat = decision_logic(y_eq, const, const)
    errors = ahat[discard:-discard] != np.roll(a, opt_delay)[discard:-discard]
    print(f"Total number of erros {np.sum(errors)} (optimal delay: {opt_delay})")
    return np.mean(errors), opt_delay


def calc_theory_ser_pam(constellation_order, EsN0_db):
    # Theorertical SER in a PAM constellation
    # Taken from Holger Krener Iversen's thesis
    snr_lin = 10 ** (EsN0_db / 10.0)
    qx = norm.sf(np.sqrt(snr_lin))  # complementary cumulative pdf of standard gaussain
    return 2 * (constellation_order - 1) / constellation_order * qx


def calc_theory_esn0_pam(constellation_order, ser):
    # Theorertical EsN0 db achieving specified SER in a PAM constellation
    # Inverse of above function
    esn0_lin = norm.isf(constellation_order / (2 * (constellation_order - 1)) * ser)
    return 20 * np.log10(esn0_lin)


def decision_logic(xhat, syms, symbol_centers=None):
    # function that interpolates to the constellation
    # assumes xhat and syms are 1D arrays
    if symbol_centers is None:
        symbol_centers = syms
    absdiff = np.abs(xhat[:, np.newaxis] - symbol_centers[np.newaxis, :])
    min_indices = np.argmin(absdiff, axis=1)
    assert (len(min_indices) == len(xhat))
    return syms[min_indices]


def calculate_confusion_matrix(ahat, a):
    assert (np.all(np.unique(ahat) == np.unique(a)))
    unique_symbols, true_symbols = np.unique(a, return_inverse=True)
    unique_symbols2, pred_symbols = np.unique(ahat, return_inverse=True)

    N = len(unique_symbols)
    conf_mat = np.zeros((N, N), dtype=np.int16)

    for true_label, predicted_label in zip(true_symbols, pred_symbols):
        conf_mat[true_label, predicted_label] += 1

    return conf_mat, unique_symbols, unique_symbols2


def decision_logic_torch(xhat, syms):
    import torch
    # DEBUG function for torch
    absdiff = torch.abs(xhat[:, None] - syms[None, :])
    min_indices = torch.argmin(absdiff, axis=1)
    return syms[min_indices]
