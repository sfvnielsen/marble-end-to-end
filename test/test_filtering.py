"""
    Test some of the filtering operations
"""

import torch

from lib.filtering import FIRfilter
from lib.equalization import SecondOrderVolterraSeries


def test_fir_forward_batched():
    sps = 4
    n_samples = sps * 10000
    seed = 12355
    n_lags = 25

    # Generate a white-noise signal
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples,), generator=g)

    # Generate random parameters to Volterra kernel
    h = torch.randn((n_lags,), generator=g).numpy()
    firfilter = FIRfilter(h, stride=sps, dtype=x.dtype)

    # Run forward pass
    with torch.no_grad():
        yf = firfilter.forward(x)
        yfb = firfilter.forward_batched(x)

    # Test if forward is equal to forward_batched
    assert torch.allclose(yf, yfb)


def test_volterra_forward_batched():
    sps = 4
    n_samples = sps * 10000
    seed = 12355
    n_lags1 = 25
    n_lags2 = 13

    # Generate a white-noise signal
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples,), generator=g)

    # Generate random parameters to Volterra kernel
    h = torch.randn((n_lags1,), generator=g)
    h2 = torch.randn((n_lags2, n_lags2), generator=g)
    h2 = h2 + h2.T

    volseries = SecondOrderVolterraSeries(n_lags1=n_lags1, n_lags2=n_lags2,
                                          samples_per_symbol=sps, dtype=x.dtype)
    volseries.kernel1.data = h
    volseries.kernel2.data = h2

    # Run forward pass
    with torch.no_grad():
        yf = volseries.forward(x)
        yfb = volseries.forward_batched(x)

    # Test if forward is equal to forward_batched
    assert torch.allclose(yf, yfb)
