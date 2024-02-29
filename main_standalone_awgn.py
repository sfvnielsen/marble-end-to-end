"""
    Standalone script for optimizing the AWGN channel
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.functional import convolve
from commpy.filters import rrcosfilter

from lib.utility import calc_ser_pam, calc_theory_ser_pam


class FIRfilter(torch.nn.Module):
    def __init__(self, filter_weights, stride=1, trainable=False, dtype=torch.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        torch_filter = torch.from_numpy(np.copy(filter_weights))
        self.filter_length = len(filter_weights)
        # self.padding = ((self.filter_length - 1) // 2, (self.filter_length - self.filter_length % 2) // 2)
        self.weights = torch.nn.Parameter(torch_filter, requires_grad=trainable)
        self.trainalbe = trainable
        self.stride = stride

    def forward(self, x):
        # input x assumed to be [timesteps,]
        return convolve(x, self.weights, mode='same')[::self.stride]

    def normalize_filter(self):
        # Calculate L2 norm and divide on the filter
        with torch.no_grad():
            self.weights /= torch.linalg.norm(self.weights)


def system_forward(symbols_up, tx_filter, rx_filter, noise_std, constellation_scale):
    # Input is assumed to be upsampled sybmols
    # Apply pulse shaper
    x = tx_filter.forward(symbols_up)

    # Add white noise
    y = x + noise_std * torch.randn(x.shape)

    # Apply rx filter
    rx_filter_out = rx_filter.forward(y)

    # Rescale to constellation (if self.normalization_after_tx is set)
    rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * constellation_scale

    return rx_filter_out


if __name__ == "__main__":

    # Simulation parameters
    symbol_rate = int(10e6)
    sps = 4  # samples pr. symbol
    esn0_db = 8.0  # symbol energy SNR
    n_symbols_train = int(1e6)
    n_symbols_test = int(1e5)
    rx_filter_length = 55
    tx_filter_length = 55
    pam_constellation = torch.Tensor([-3, -1, 1, 3])
    constellation_scale = torch.sqrt(torch.mean(torch.square(pam_constellation)))
    rrc_rolloff = 0.5
    dtype = 'float32'

    # Optimization parameters
    batch_size = 1000  # number of symbols
    learning_rate = 1e-4

    # Derived parameters
    sym_length = 1 / symbol_rate
    Ts = sym_length / sps
    Fs = 1 / Ts
    noise_std = np.sqrt(1.0 / (2 * 10 ** (esn0_db / 10)))
    batch_print_interval = 10000 / batch_size
    discard_n_syms = (rx_filter_length + tx_filter_length) // sps

    # Create torch objects
    tx_filter_init = np.zeros((tx_filter_length,), dtype=dtype)
    tx_filter_init[tx_filter_length//2 - 1] = 1.0
    tx_filter = FIRfilter(filter_weights=tx_filter_init, trainable=True)

    __, rx_filter_init = rrcosfilter(rx_filter_length, rrc_rolloff, sym_length, Fs)
    rx_filter_init = rx_filter_init.astype(dtype)
    rx_filter = FIRfilter(rx_filter_init, stride=sps, trainable=False)

    optimizer = torch.optim.Adam([{"params": tx_filter.parameters()}], lr=learning_rate)

    for b in range(n_symbols_train // batch_size):
        # Zero gradients
        tx_filter.zero_grad()
        rx_filter.zero_grad()

        # Sample a new batch of symbols
        syms = pam_constellation[torch.randint(len(pam_constellation), (batch_size,))]
        syms_up = torch.zeros((batch_size * sps,))
        syms_up[::sps] = syms

        # Run upsampled symbols through system forward model - return symbols at Rx
        rx_out = system_forward(syms_up, tx_filter, rx_filter, noise_std, constellation_scale)

        # Calculate loss - discard some symbols in beginning and 
        # end of batch to remove boundary effects of convolution
        loss = torch.mean(torch.square(rx_out[discard_n_syms:-discard_n_syms] - syms[discard_n_syms:-discard_n_syms]))

        # Update model using backpropagation
        loss.backward()
        optimizer.step()

        # Normalize the filter in each step (to no arbitrarily increase the power)
        tx_filter.normalize_filter()

        if b % batch_print_interval == 0:
            print(f"Batch {b} (# symbols {b * batch_size:.2e}) - Loss: {loss.item():.3f}")

    # Create a new set of symbols to evaluate the SER on
    test_syms = pam_constellation[torch.randint(len(pam_constellation), (n_symbols_test,))]
    test_syms_up = torch.zeros((n_symbols_test * sps,))
    test_syms_up[::sps] = test_syms

    # Run the system model (without gradients) to evaluate system performance
    with torch.no_grad():
        rx_test_out = system_forward(test_syms_up, tx_filter, rx_filter, noise_std, constellation_scale)

    # Calculate symbol error rate (including theoretical AWGN estimate - based on SNR)
    ser, __ = calc_ser_pam(rx_test_out.numpy(), test_syms.numpy(), discard=50)
    ser_theoretical = calc_theory_ser_pam(len(pam_constellation), esn0_db)

    # Plot the distribution
    fig, ax = plt.subplots()
    ax.hist(rx_test_out.numpy(), bins=1000)
    ax.set_title(f"SER: {ser:.3e} (Theoretical: {ser_theoretical:.3e})")

    plt.show()