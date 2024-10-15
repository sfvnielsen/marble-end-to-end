"""
    Collection of devices used in transmission systems
"""

import torch
import numpy as np
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from scipy.interpolate import CubicSpline
from scipy.optimize import newton

from .filtering import BesselFilter, MultiChannelBesselFilter, AllPassFilter, LowPassFIR


class IdealLinearModulator(object):
    """
        Ideal linear modulator

    """
    def __init__(self, laser_power_dbm):
        # Parse input arguments
        self.laser_power = 10 ** (laser_power_dbm / 10) * 1e-3  # [Watt]
        self.Plaunch_dbm = None

        # Define range for input - will be clamped on call
        self.input_range = (0.0, None)


    def get_launch_power_dbm(self):
        if self.Plaunch_dbm is None:
            raise Exception("Launch power has not been calculated yet!")

        return self.Plaunch_dbm

    def forward(self, v):
        # Assumes that v is the input voltage is positive
        v = torch.clamp(v, *self.input_range)

        # Return transmitted field amplitude - assumes to be used together with a square-law photodetector
        y = torch.sqrt(self.laser_power * v)
        self.Plaunch_dbm = 10.0 * torch.log10(torch.mean(torch.square(y)) / 1e-3)
        return y


class MachZehnderModulator(object):
    """
        Classic MZM

        Heavily inspired by Edson Porto Silva's OptiCommPy
        https://github.com/edsonportosilva/OptiCommPy

    """
    def __init__(self, laser_power_dbm: float, vpi: float = 2.0, vb: float = -0.5) -> None:
        self.vpi = vpi
        self.vb = vb
        self.laser_power = 10 ** (laser_power_dbm / 10) * 1e-3  # [Watt]
        self.laser_amplitude = np.sqrt(self.laser_power)
        self.Plaunch_dbm = None

    def get_launch_power_dbm(self):
        if self.Plaunch_dbm is None:
            raise Exception("Launch power has not been calculated yet!")

        return self.Plaunch_dbm

    def forward(self, v):
        y = self.laser_amplitude * torch.cos(0.5 / self.vpi * (v + self.vb) * torch.pi)
        self.Plaunch_dbm = 10.0 * torch.log10(torch.mean(torch.square(y)) / 1e-3)
        return y


class ElectroAbsorptionModulator(object):
    """
        Implementation of and electro absorption modulator (EAM) as used in

        E. M. Liang and J. M. Kahn,
        “Geometric Shaping for Distortion-Limited Intensity Modulation/Direct Detection Data Center Links,”
        IEEE Photonics Journal, vol. 15, no. 6, pp. 1–17, 2023, doi: 10.1109/JPHOT.2023.3335398.

    """

    # Voltage-to-absorption curve directly from (Liang and Kahn, 2023)
    ABSORPTION_KNEE_POINTS_X = torch.flip(torch.Tensor([0.0, -0.5,  -1.0, -2.0, -3.0, -3.5,  -3.8]), (0,))  # driving voltage
    ABSORPTION_KNEE_POINTS_Y = torch.flip(torch.Tensor([0.0, 1.25,   2.5,  5.0,  9.5, 13.0,  12.5]), (0,))  # absorption in dB

    # Linear voltage-to-absorption curve
    LINEAR_ABSORPTION_KNEE_POINTS_X = torch.flip(torch.Tensor([0.0, -1.0,  -2.0, -3.0, -4.0]), (0,))  # driving voltage
    LINEAR_ABSORPTION_KNEE_POINTS_Y = torch.flip(torch.Tensor([0.0, 1.0,   2.0,  3.0,  4.0]), (0,))  # absorption in dB

    def __init__(self, laser_power_dbm,
                 linewidth_enhancement=0.0, linear_absorption=False):
        # Parse input arguments
        self.laser_power = 10 ** (laser_power_dbm / 10) * 1e-3  # [Watt]
        self.linewidth_enhancement = linewidth_enhancement  # chirp model (reasonable values in the interval [0,3])

        self.x_knee_points = self.ABSORPTION_KNEE_POINTS_X if not linear_absorption else self.LINEAR_ABSORPTION_KNEE_POINTS_X
        self.y_knee_points = self.ABSORPTION_KNEE_POINTS_Y if not linear_absorption else self.LINEAR_ABSORPTION_KNEE_POINTS_Y

        # Caculate the cubic spline coefficients based on the given knee-points
        spline_coefficients = natural_cubic_spline_coeffs(self.x_knee_points, self.y_knee_points[:, None])
        self.spline_object = NaturalCubicSpline(spline_coefficients)

        # Define range for input - will be clamped on call
        self.input_range = (-4.5, 0.0)

        # Initialize launch power
        self.Plaunch_dbm = None

    def get_launch_power_dbm(self):
        if self.Plaunch_dbm is None:
            raise Exception("Launch power has not been calculated yet!")

        return self.Plaunch_dbm

    def forward(self, v):
        # Clamp input to valid range
        v = torch.clamp(v, *self.input_range)

        # Calculate absorption from voltage
        alpha_db = self.spline_object.evaluate(v.contiguous()).squeeze()

        # Return transmitted field amplitude
        y = torch.sqrt(self.laser_power * torch.pow(10.0, -alpha_db / 10.0))

        # Calculate chirp
        chirp = self.linewidth_enhancement / 2 * torch.log(torch.square(y))

        # Log launch power
        self.Plaunch_dbm = 10.0 * torch.log10(torch.mean(torch.square(y)) / 1e-3)

        return y * torch.exp(1j * chirp)


class MyNonLinearEAM(ElectroAbsorptionModulator):
    """
        Electro absorption modulator - but with custom very non-linear absorption curve
        (cf. parent class above)
    """
    ABSORPTION_KNEE_POINTS_X = torch.flip(torch.Tensor([0.0, -0.5,  -1.0, -2.5, -3.5,  -4.0]), (0,))  # driving voltage
    ABSORPTION_KNEE_POINTS_Y = torch.flip(torch.Tensor([0.0,  0.5,   1.0,  2.5,  2.6,  2.6]), (0,))  # absorption in dB


class Photodiode(object):
    """
        pin photodiode implementation

        Heavily inspired by Edson Porto Silva's OptiCommPy
        https://github.com/edsonportosilva/OptiCommPy

        responsivity: [A/W]
        temperature: [Celcius]
        dark current: [A]
        impedance load: [Ohm]
        bandwidth: [Hz]
        Fs: [Hz]

    """
    ELECTRON_CHARGE = 1.60217662e-19
    BOLTZMANN = 1.3806485e-23

    def __init__(self, bandwidth, Fs, sps, responsivity=1.0, temperature=20.0, dark_current=10e-9,
                 impedance_load=50.0) -> None:

        # Add constructor arguments as attributes
        self.responsivity = responsivity
        self.temperature = temperature
        self.dark_current = dark_current
        self.impedance_load = impedance_load
        self.bandwidth = bandwidth
        self.Fs = Fs
        self.sps = sps  # used for calculation energy-pr-symbol

        # Calculation of thermal noise std
        var_thermal = 4 * self.BOLTZMANN * (self.temperature + 273.15) * \
            self.bandwidth / self.impedance_load

        self.thermal_noise_std = np.sqrt(self.Fs * (var_thermal / (2 * self.bandwidth)))
        self.responsivity = responsivity
        self.dark_current = dark_current

        self.sps = sps
        self.Es = None  # energy pr. symbol after detection
        self.Prec = None  # received power [dBm]
        self.total_noise_variance = None

    def get_thermal_noise_std(self):
        return self.thermal_noise_std
    
    def get_total_noise_variance(self):
        if self.total_noise_variance is None:
            raise Exception("Total noise variance has not been calculated yet!")
        return self.total_noise_variance.numpy()

    def get_received_power_dbm(self):
        if self.Prec is None:
            raise Exception("Received power has not been calculated yet!")
        return self.Prec

    def forward(self, x):
        # Square law detection of input
        x2 = torch.real(torch.square(torch.abs(x)))

        # Generate thermal noise
        thermal_noise = self.thermal_noise_std * torch.randn_like(x2)

        # Calculate shot_noise variance based on avg. signal power
        var_shot = 2 * self.ELECTRON_CHARGE * (self.responsivity * torch.mean(torch.square(torch.real(torch.absolute(x)))) +
                                               self.dark_current) * self.bandwidth

        shot_noise = torch.sqrt(self.Fs * (var_shot / (2 * self.bandwidth))) * torch.randn_like(x2)

        # Update energy pr. symbol and received power before addition of noise
        self.Prec = 10.0 * torch.log10(torch.mean(x2) / 1e-3)
        self.Es = torch.mean(torch.sum(torch.square(torch.reshape((x2 - x2.mean())[0:len(x2)//self.sps * self.sps], (-1, self.sps))), dim=1))
        self.total_noise_variance = self.thermal_noise_std**2 + self.Fs * (var_shot / (2 * self.bandwidth))

        return x2 + thermal_noise + shot_noise


def quantize(signal, bits):
    """

        Written almost entirely by Microsoft Copilot
    """
    max_int = 2**bits - 1

    # Normalize the float array to the range [0, 1]
    normalized = (signal - torch.min(signal)) / (torch.max(signal) - torch.min(signal))

    # Scale to the quantization levels and round to nearest integer
    quantized = torch.round(normalized * max_int)

    # Map back to the original range
    dequantized = quantized * (torch.max(signal) - torch.min(signal)) / max_int + torch.min(signal)

    return dequantized


class DigitalToAnalogConverter(torch.nn.Module):
    """
        DAC with bandwidth limitation modeled by a Bessel filter
    """
    def __init__(self, bwl_cutoff, peak_to_peak_voltage, peak_to_peak_constellation: float | str,
                 fs, bias_voltage: float | str ='negative_vpp', bit_resolution=None, lpf_order=5,
                 learnable_normalization=False, learnable_bias=False, multi_channel: bool=False,
                 filter_type='bessel', clamp_min=-0.5, clamp_max=0.5,
                 dtype=torch.float64, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set attributes of DAC
        self.learnable_normalization = learnable_normalization
        self.learnable_bias = learnable_bias
        self.v_pp = peak_to_peak_voltage
        self.pp_const = peak_to_peak_constellation  # distance between largest and smallest constellation point
        self.voltage_norm_funcp = self._vol_norm_const
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
        self._clamp_funcp = self._clamp_voltage_std
        if self.clamp_min is None and self.clamp_max is None:
            self._clamp_funcp = self._bypass_clamp

        norm_init = 1.0
        bias_init = 0.0

        if isinstance(bias_voltage, str) and bias_voltage == "positive_vpp":
            # Move the voltages to positive domain with max value Vpp (min = 0)
            bias_init = self.v_pp/2
        elif isinstance(bias_voltage, str) and bias_voltage == "negative_vpp":
            # Move the voltages to negative domain with max value 0 (based on the Vpp)
            bias_init = -self.v_pp/2
        elif isinstance(bias_voltage, float):
            bias_init = bias_voltage
        else:
            print(f"Unknown voltage bias type {bias_voltage}. Using a bias of 0.0")
        
        self.v_bias = torch.nn.Parameter(torch.scalar_tensor(bias_init, dtype=dtype), requires_grad=self.learnable_bias)

        if isinstance(self.pp_const, str) and self.pp_const == "minmax":
            self.voltage_norm_funcp = self._vol_norm_minmax
            print(f"DAC: minmax mode. WARNING! This is not nicely differentiable.")
        elif isinstance(self.pp_const, float):
            self.voltage_norm_funcp = self._vol_norm_const
            norm_init = 1 / self.pp_const
        else:
            raise Exception(f"Failed parsing 'peak_to_peak_constellation' argument ({peak_to_peak_constellation})")

        self.normalization = torch.nn.Parameter(torch.scalar_tensor(norm_init, dtype=dtype), requires_grad=self.learnable_normalization)
        self.bit_resolution = bit_resolution

        # Initialize bessel filter
        self.lpf = AllPassFilter()
        if bwl_cutoff is not None:
            if filter_type.lower() == 'bessel':
                if multi_channel:
                    self.lpf = MultiChannelBesselFilter(bessel_order=lpf_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)
                else:
                    self.lpf = BesselFilter(bessel_order=lpf_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)
            elif filter_type.lower() == 'fir':
                if multi_channel:
                    raise NotImplementedError(f"Have not implemented multi-channel version of FIR yet...")
                else:
                    self.lpf = LowPassFIR(num_taps=lpf_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

    def clamp_voltage(self, x: torch.TensorType) -> torch.TensorType:
        return self._clamp_funcp(x)

    def _clamp_voltage_std(self, x):
        return torch.clamp(x, self.clamp_min, self.clamp_max)
    
    def _bypass_clamp(self, x):
        return x

    def set_bitres(self, new_bitres):
        assert isinstance(new_bitres, int) or new_bitres is None
        self.bit_resolution = new_bitres

    def get_sample_delay(self):
        return self.lpf.get_sample_delay()

    def get_lpf_filter(self):
        return self.lpf.get_filters()

    def voltage_normalization(self, x: torch.TensorType) -> torch.TensorType:
        return self.voltage_norm_funcp(x)

    def _vol_norm_minmax(self, x: torch.TensorType):
        return (x - x.min()) / (x.max() - x.min()) - 0.5

    def _vol_norm_const(self, x: torch.TensorType):
        # x is assumed to have self.pp_const between largest and smallest constellation point
        return x * self.normalization

    def forward(self, x):
        # Map digital signal to a voltage - clamped to [-Vpp/2, Vpp/2]
        v = self.v_pp * self.clamp_voltage(self.voltage_normalization(x))

        # Run lpf
        v_lp = self.lpf.forward(v)

        return v_lp + self.v_bias

    def eval(self, x):
        # Map digital signal to a voltage
        v = self.v_pp * self.clamp_voltage(self.voltage_normalization(x))

        if self.bit_resolution:
            v = quantize(v, self.bit_resolution)

        # Run lpf
        v_lp = self.lpf.forward_numpy(v)

        return v_lp + self.v_bias


class AnalogToDigitalConverter(object):
    """
        ADC with bandwidth limitation modeled by a low-pass filter
    """
    def __init__(self, bwl_cutoff, fs, bit_resolution=None, lpf_order=5, 
                 filter_type='bessel', dtype=torch.float64) -> None:
        # Set attributes
        self.bit_resolution = bit_resolution
        self.dtype = dtype

        # Initialize bessel filter
        self.lpf = AllPassFilter()
        if bwl_cutoff is not None and filter_type == 'bessel':
            self.lpf = BesselFilter(bessel_order=lpf_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)
        elif bwl_cutoff is not None and filter_type == 'fir':
            self.lpf = LowPassFIR(num_taps=lpf_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

    def set_bitres(self, new_bitres):
        assert isinstance(new_bitres, int) or new_bitres is None
        self.bit_resolution = new_bitres

    def get_sample_delay(self):
        return self.lpf.get_sample_delay()

    def get_lpf_filter(self):
        return self.lpf.get_filters()

    def forward(self, v):
        # Run lpf
        x_lp = self.lpf.forward(v)

        return x_lp

    def eval(self, v):
        # Run lpf
        x_lp = self.lpf.forward_numpy(v)

        # Quantize if specfied
        if self.bit_resolution:
            x_lp = quantize(x_lp, self.bit_resolution)

        return x_lp