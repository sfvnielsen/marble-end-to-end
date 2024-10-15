"""
    Library of channel models - all implemented in torch without any optimizable parameters
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.fft import fft, ifft, fftfreq, fftshift
from copy import deepcopy

from .filtering import FIRfilter, MultiChannelFIRfilter

class SingleModeFiber(object):
    """
        SMF with chromatic dispersion

        Heavily inspired by Edson Porto Silva's OptiCommPy
        https://github.com/edsonportosilva/OptiCommPy

        Inputs to constructor.

        fiber_length in [km]
        attenutation in [dB / km]
        carrier_wavelength in [nm]
        Fs sampling frequency in Hz

        Dispersion is specified either as

        dispersion_parameter in [ps / (nm * km)]

        or

        dispersion_slope in [ps / (nm^2 * km)]
        zero_dispersion_wavelength in [nm]

        References:
        E. M. Liang and J. M. Kahn, “Geometric Shaping for Distortion-Limited Intensity Modulation/Direct Detection Data Center Links,”
        IEEE Photonics Journal, vol. 15, no. 6, pp. 1–17, 2023, doi: 10.1109/JPHOT.2023.3335398.


    """
    SPEED_OF_LIGHT = 299792458  # [m / s]

    def __init__(self, fiber_length, attenuation, carrier_wavelength, Fs,
                 dispersion_parameter=None, dispersion_slope=None, zero_dispersion_wavelength=None) -> None:

        # Add properties from constructor
        self.fiber_length = fiber_length
        self.attenuation = attenuation
        self.carrier_wavelength = carrier_wavelength
        self.Fs = Fs

        # Check dispersion arguments
        if dispersion_parameter is None and ((dispersion_slope is None) or (zero_dispersion_wavelength is None)):
            raise ValueError("Dispersion parameters cannot all be None...")

        # Calculate parameters based on inputs
        self.dispersion_parameter = dispersion_parameter
        if self.dispersion_parameter is None:
            # Convert zero_dispersion_wavelength and carrier_wavelength to dispersion parameter (cf. Liang and Kahn, 2023)
            self.dispersion_parameter = dispersion_slope * (carrier_wavelength - zero_dispersion_wavelength**4 / carrier_wavelength**3)  # [ps / (nm * km)]

        self.alpha = self.attenuation / (10 * np.log10(np.exp(1.0)))  # [1/km]
        carrier_wavelength_in_km = self.carrier_wavelength / (1e9 * 1e3)  # from nm to km
        self.beta2 = -(self.dispersion_parameter * (carrier_wavelength_in_km)**2) / (2 * torch.pi * self.SPEED_OF_LIGHT / 1e3)  # [s^2/km]

    # FIXME: Add symbol delay?

    def _calculate_fq_filter(self, omega):
        return torch.exp(-self.alpha/2 * self.fiber_length + 1j * (self.beta2 / 2) * (omega**2) * self.fiber_length)

    def get_fq_filter(self, signal_length):
        freq = self.Fs * fftfreq(signal_length)
        return fftshift(freq).numpy(), self._calculate_fq_filter(freq).detach().numpy()

    def forward(self, x: torch.Tensor):
        # FIXME: Add zero padding before doing fft?
        Nfft = len(x)
        omega = 2 * torch.pi * self.Fs * fftfreq(Nfft)
        xo = ifft(fft(x) * self._calculate_fq_filter(omega))
        return xo


class WienerHammersteinChannel(torch.nn.Module):
    """
        Simple Wiener-Hammerstein system.
        Two FIR filters with a polynomial sandwich between.
        Polynomial order is fixed to three
    """
    def __init__(self, n_taps1: int, n_taps2: int, dtype=torch.float64):
        super().__init__()

        # Initialize FIR filters
        h1 = np.zeros((n_taps1,))
        h1[n_taps1 // 2] = 1.0  # dirac initialization
        self.fir1 = FIRfilter(h1, stride=1, trainable=True, dtype=dtype)

        h2 = np.zeros((n_taps2,))
        h2[n_taps2 // 2] = 1.0  # dirac initialization
        self.fir2 = FIRfilter(h2, stride=1, trainable=True, dtype=dtype)

        # Polynomial coefficients - index 0 is bias
        self.poly_coeffs = torch.nn.Parameter(torch.Tensor([0.0, 1.0, 0.0, 0.0]), requires_grad=True).to(dtype)

    def forward(self, x):
        y1 = self.fir1.forward(x)
        z = self.poly_coeffs[0] + self.poly_coeffs[1] * y1 + self.poly_coeffs[2] * y1**2 + self.poly_coeffs[3] * y1**3
        return self.fir2.forward(z)

    def discard_samples(self):
        return self.fir1.filter_length // 2 + self.fir2.filter_length // 2


def create_linear_layer(hidden_units, dropout=None):
        layers = [torch.nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                  torch.nn.ReLU()]
        if dropout:
            layers.append(torch.nn.Dropout(dropout))
        return torch.nn.Sequential(*layers)


class HammersteinNN(torch.nn.Module):
    """
        CNN/FFN mimicking a Hammerstein structure

        Fully-connected layers learn the non-linearity
        Afterwards FIR filter is applied to learn the ISI
    """
    def __init__(self, n_lags, n_hidden_units, n_hidden_layers, dtype=torch.double,
                 torch_device=torch.device('cpu'), **kwargs) -> None:
        super().__init__(**kwargs)

        # Set properties of class
        self.dtype = dtype
        self.torch_device = torch_device

        # Initialize kernels (FIR filters pre- and post non-linearity)
        assert (n_lags + 1) % 2 == 0
        kernel_init = np.zeros((n_lags, ))
        kernel_init[n_lags // 2] = 1.0  # dirac initialization
        self.fir_filter = torch.nn.Parameter(torch.from_numpy(kernel_init).to(dtype).to(self.torch_device))
        self.padding = n_lags // 2

        # Initialize the fully-connected NN
        self.linear_explode = torch.nn.Linear(in_features=1, out_features=n_hidden_units).to(dtype).to(self.torch_device)
        self.hidden_layers = torch.nn.ModuleList([create_linear_layer(n_hidden_units).to(dtype).to(self.torch_device) for __ in range(n_hidden_layers)])
        self.linear_join = torch.nn.Linear(in_features=n_hidden_units, out_features=1).to(dtype).to(self.torch_device)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # 'Explode' size to the hidden dimension
        z = self.linear_explode.forward(x[:, None])
        
        # Apply the FCs
        for layer in self.hidden_layers:
            z = layer.forward(z)
        
        # Reduce dimensionality
        z2 = self.linear_join.forward(z).squeeze()

        # Pad and run FIR filter
        z2pad = torch.concatenate((torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device),
                                  z2,
                                  torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device)))

        y = F.conv1d(z2pad[None, None, :], self.fir_filter[None, None, :]).squeeze()

        return y

    def discard_samples(self):
        return len(self.fir_filter) // 2
    
class WienerNN(torch.nn.Module):
    """
        CNN/FFN mimicking a Wiener structure

        FIR filter is applied to learn the ISI
        Afterwards fully-connected layers learn the non-linearity
    """
    def __init__(self, n_lags, n_hidden_units, n_hidden_layers, dtype=torch.double,
                 torch_device=torch.device('cpu'), **kwargs) -> None:
        super().__init__(**kwargs)

        # Set properties of class
        self.dtype = dtype
        self.torch_device = torch_device

        # Initialize kernels (FIR filters pre- and post non-linearity)
        assert (n_lags + 1) % 2 == 0
        kernel_init = np.zeros((n_lags, ))
        kernel_init[n_lags // 2] = 1.0  # dirac initialization
        self.fir_filter = torch.nn.Parameter(torch.from_numpy(kernel_init).to(dtype).to(self.torch_device))
        self.padding = n_lags // 2

        # Initialize the fully-connected NN
        self.linear_explode = torch.nn.Linear(in_features=1, out_features=n_hidden_units).to(dtype).to(self.torch_device)
        self.hidden_layers = torch.nn.ModuleList([create_linear_layer(n_hidden_units).to(dtype).to(self.torch_device) for __ in range(n_hidden_layers)])
        self.linear_join = torch.nn.Linear(in_features=n_hidden_units, out_features=1).to(dtype).to(self.torch_device)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # Apply FIR
        xpad = torch.concatenate((torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device),
                                  x,
                                  torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device)))
        z1 = F.conv1d(xpad[None, None, :], self.fir_filter[None, None, :]).squeeze()

        # Apply explosion layer
        z = self.linear_explode.forward(z1[:, None])
        
        # Apply the FCs
        for layer in self.hidden_layers:
            z = layer.forward(z)
        
        # Reduce dimensionality
        y = self.linear_join.forward(z).squeeze()

        return y

    def discard_samples(self):
        return len(self.fir_filter) // 2

class WienerHammersteinNN(torch.nn.Module):
    """
        CNN/FFN mimicking a Wiener-Hammerstein structure

        FIR filter is applied to learn the ISI
        Fully-connected layers learn the non-linearity
        Finally, FIR filter is applied again
    """
    def __init__(self, n_lags, n_hidden_units, n_hidden_layers, dtype=torch.double,
                 torch_device=torch.device('cpu'), **kwargs) -> None:
        super().__init__(**kwargs)

        # Set properties of class
        self.dtype = dtype
        self.torch_device = torch_device

        # Initialize kernels (FIR filters pre- and post non-linearity)
        assert (n_lags + 1) % 2 == 0
        kernel_init = np.zeros((n_lags, ))
        kernel_init[n_lags // 2] = 1.0  # dirac initialization
        self.fir_filter1 = torch.nn.Parameter(torch.from_numpy(kernel_init).to(dtype).to(self.torch_device))
        self.fir_filter2 = torch.nn.Parameter(torch.from_numpy(kernel_init).to(dtype).to(self.torch_device))
        self.padding = n_lags // 2

        # Initialize the fully-connected NN
        self.linear_explode = torch.nn.Linear(in_features=1, out_features=n_hidden_units).to(dtype).to(self.torch_device)
        self.hidden_layers = torch.nn.ModuleList([create_linear_layer(n_hidden_units).to(dtype).to(self.torch_device) for __ in range(n_hidden_layers)])
        self.linear_join = torch.nn.Linear(in_features=n_hidden_units, out_features=1).to(dtype).to(self.torch_device)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # Apply FIR
        xpad = torch.concatenate((torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device),
                                  x,
                                  torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device)))
        z1 = F.conv1d(xpad[None, None, :], self.fir_filter1[None, None, :]).squeeze()

        # Apply explosion layer
        z = self.linear_explode.forward(z1[:, None])
        
        # Apply the FCs
        for layer in self.hidden_layers:
            z = layer.forward(z)
        
        # Reduce dimensionality
        z2 = self.linear_join.forward(z).squeeze()

        # Pad and run second FIR filter
        z2pad = torch.concatenate((torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device),
                                  z2,
                                  torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device)))

        y = F.conv1d(z2pad[None, None, :], self.fir_filter2[None, None, :]).squeeze()

        return y

    def discard_samples(self):
        return len(self.fir_filter1) // 2 + len(self.fir_filter2) // 2


class CNN(torch.nn.Module):
    """
        CNN with multiple filters and a standard fully connected network at the end
    """
    def __init__(self, n_lags, n_hidden_units, n_hidden_layers, dtype=torch.double,
                 torch_device=torch.device('cpu'), **kwargs) -> None:
        super().__init__(**kwargs)

        # Set properties of class
        self.dtype = dtype
        self.torch_device = torch_device

        # Initialize kernels for CNN part
        assert (n_lags + 1) % 2 == 0
        self.n_lags = n_lags
        self.cnn = torch.nn.Conv1d(in_channels=1, out_channels=n_hidden_units, kernel_size=n_lags,
                                   padding=n_lags // 2, dtype=dtype, device=torch_device)

        # Initialize the fully-connected NN
        self.hidden_layers = torch.nn.ModuleList([create_linear_layer(n_hidden_units).to(dtype).to(self.torch_device) for __ in range(n_hidden_layers)])
        self.linear_join = torch.nn.Linear(in_features=n_hidden_units, out_features=1).to(dtype).to(self.torch_device)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # Apply CNN
        z = self.cnn.forward(x[None, None, :]).squeeze().T

        # Apply the FCs
        for layer in self.hidden_layers:
            z = layer.forward(z)
        
        # Reduce dimensionality
        y = self.linear_join.forward(z).squeeze()

        return y

    def discard_samples(self):
        return self.n_lags



class SurrogateChannel(torch.nn.Module):
    """
        Surrogate channel that estimates the channel response through a
        non-differentiable channel of interest.
        This enables backpropagation back to any transmitter parameters.
    """
    def __init__(self, multi_channel: bool, **kwargs):
        super().__init__()
        local_kwarg_copy = deepcopy(kwargs)
        self.surrogate_type = local_kwarg_copy.pop('type')
        self.multi_channel = multi_channel
        self.bias = torch.nn.Parameter(torch.scalar_tensor(0.0, dtype=torch.double, requires_grad=True))

        if self.surrogate_type.lower() == 'fir':
            self.filter_length = local_kwarg_copy['n_taps']
            h = np.zeros((local_kwarg_copy['n_taps'],))
            h[local_kwarg_copy['n_taps'] // 2] = 1.0  # dirac initialization
            if self.multi_channel:
                # FIXME: Verify that this works in WDM setup
                self.channel_model = MultiChannelFIRfilter(h, stride=1, trainable=True)
            else:
                self.channel_model = FIRfilter(h, stride=1, trainable=True)
        elif self.surrogate_type.lower() == 'wh':
            if self.multi_channel:
                # FIXME: Implement multi-channel version of WienerHammerstein channel.
                raise NotImplementedError
            else:
                self.channel_model = WienerHammersteinChannel(**local_kwarg_copy)
        elif self.surrogate_type.lower() == 'wiener_nn':
            if self.multi_channel:
                # FIXME: Implement multi-channel version of this channel estimator.
                raise NotImplementedError
            else:
                self.channel_model = WienerNN(**local_kwarg_copy)
        elif self.surrogate_type.lower() == 'hammerstein_nn':
            if self.multi_channel:
                # FIXME: Implement multi-channel version of this channel estimator.
                raise NotImplementedError
            else:
                self.channel_model = HammersteinNN(**local_kwarg_copy)
        elif self.surrogate_type.lower() == 'wiener_hammerstein_nn':
            if self.multi_channel:
                # FIXME: Implement multi-channel version of this channel estimator.
                raise NotImplementedError
            else:
                self.channel_model = WienerHammersteinNN(**local_kwarg_copy)
        elif self.surrogate_type.lower() == 'cnn':
            if self.multi_channel:
                # FIXME: Implement multi-channel version of this channel estimator.
                raise NotImplementedError
            else:
                self.channel_model = CNN(**local_kwarg_copy)
        else:
            raise ValueError(f"Unknown surrogate channel model type: {self.surrogate_type}")

    def forward(self, x):
        return self.channel_model.forward(x) + self.bias

    def get_samples_discard(self):
        if self.surrogate_type.lower() == 'fir':
            return self.filter_length // 2
        elif self.surrogate_type.lower() == 'wh' or \
             self.surrogate_type.lower() == 'wiener_nn' or \
             self.surrogate_type.lower() == 'hammerstein_nn' or \
             self.surrogate_type.lower() == 'cnn':
            return self.channel_model.discard_samples()
        else:
            print(f"WARNING! Unknown channel model...")
            return 0
