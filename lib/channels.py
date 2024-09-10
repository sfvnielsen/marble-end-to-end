"""
    Library of channel models - all implemented in torch without any optimizable parameters
"""

import torch
import numpy as np
from torch.fft import fft, ifft, fftfreq, fftshift

from .filtering import FIRfilter

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


class SurrogateChannel(torch.nn.Module):
    def __init__(self, n_taps, **kwargs):
        super().__init__(**kwargs)
        self.filter_length = n_taps
        h = np.zeros((n_taps,))
        h[n_taps // 2] = 1.0  # dirac initialization
        self.channel_model = FIRfilter(h, stride=1, trainable=True)
    
    def forward(self, x):
        return self.channel_model.forward(x)
    
    def get_samples_discard(self):
        return self.filter_length // 2
