import numpy as np
from scipy.stats import norm


def find_delay(ahat, a, delay_order, sublength=1000):
    """ Find delay that maximizes correlation between real-parts
    """
    assert (len(ahat) > (sublength + delay_order) and len(a) > (sublength + delay_order))
    corr = np.zeros((delay_order,))
    for i in range(delay_order):
        corr[i] = np.dot(np.real(ahat[delay_order:(sublength + delay_order)]), np.real(np.roll(a, i)[delay_order:(sublength + delay_order)])) / sublength
    return np.argmax(np.absolute(corr))


def find_max_variance_sample(y, sps):
    nsyms = len(y) // sps  # truncate to an integer num symbols
    yr = np.reshape(y[0:nsyms*sps], (-1, sps))
    var = np.var(yr, axis=0)
    max_variance_sample = np.argmax(var)
    return max_variance_sample


def calc_ser_pam(y_eq, a, delay_order=30, sublength=1000, discard=10):
    assert len(y_eq) == len(a)
    opt_delay = 0
    if delay_order != 0:
        opt_delay = find_delay(y_eq, a, delay_order=delay_order, sublength=sublength)
    const = np.unique(a)
    ahat = decision_logic(y_eq, const, const)
    errors = ahat[discard:-discard] != np.roll(a, opt_delay)[discard:-discard]
    if np.sum(errors) < 10:
        print(f"WARNING!: Total number of errors was {np.sum(errors)} in SER calculation.")
    return np.mean(errors), opt_delay


def calc_theory_ser_pam(constellation_order, EsN0_db):
    # Theorertical SER in a PAM constellation
    # Taken from Holger Krener Iversen's thesis
    snr_lin = 10 ** (EsN0_db / 10.0)
    qx = norm.sf(np.sqrt(snr_lin))  # complementary cumulative pdf of standard gaussain
    return 2 * (constellation_order - 1) / constellation_order * qx


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
