"""
    Collection of devices used in transmission systems
"""

import torch
import numpy as np
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from scipy.interpolate import CubicSpline
from scipy.optimize import newton

from .filtering import BesselFilter, AllPassFilter


class IdealLinearModulator(object):
    """
        Ideal linear modulator
        
    """
    def __init__(self, laser_power_dbm):
        # Parse input arguments
        self.laser_power = 10 ** (laser_power_dbm / 10) * 1e-3  # [Watt]
        self.Plaunch_dbm = None

    def get_launch_power_dbm(self):
        if self.Plaunch_dbm is None:
            raise Exception("Launch power has not been calculated yet!")

        return self.Plaunch_dbm

    def forward(self, v):
        # Assumes that v is the input voltage (normalized to the [0, 1] range)
        # Return transmitted field amplitude - assumes to be used together with a square-law photodetector
        y = torch.sqrt(self.laser_power * v)
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

    def __init__(self, insertion_loss, pp_voltage, bias_voltage, laser_power_dbm,
                 linewidth_enhancement=0.0, linear_absorption=False):
        # Parse input arguments
        self.insertion_loss = insertion_loss
        self.pp_voltage = pp_voltage
        self.bias_voltage = bias_voltage
        self.laser_power = 10 ** (laser_power_dbm / 10) * 1e-3  # [Watt]
        self.linewidth_enhancement = linewidth_enhancement  # chirp model (reasonable values in the interval [0,3])

        self.x_knee_points = self.ABSORPTION_KNEE_POINTS_X if not linear_absorption else self.LINEAR_ABSORPTION_KNEE_POINTS_X
        self.y_knee_points = self.ABSORPTION_KNEE_POINTS_Y if not linear_absorption else self.LINEAR_ABSORPTION_KNEE_POINTS_Y

        # Caculate the cubic spline coefficients based on the given knee-points
        spline_coefficients = natural_cubic_spline_coeffs(self.x_knee_points, self.y_knee_points[:, None])
        self.spline_object = NaturalCubicSpline(spline_coefficients)

        # Calculate voltage corresponding to given insertion loss
        cubicspline = CubicSpline(self.x_knee_points.numpy(), self.y_knee_points.numpy())
        self.voltage_insertion_loss = newton(lambda x: cubicspline(x) - self.insertion_loss, x0=-1.0)
        self.voltage_min = self.bias_voltage + self.pp_voltage / 2 + self.voltage_insertion_loss

        # Input to forward is assumed to be a digital signal - convert to voltages using dac_min_max
        # (typically based on constellation values)
        minmaxeval = self.spline_object.evaluate(torch.Tensor([1, 0]) * self.pp_voltage + self.bias_voltage + self.voltage_insertion_loss).squeeze().numpy()
        self.ex_ratio = minmaxeval[1] - minmaxeval[0]

        # Initialize launch power
        self.Plaunch_dbm = None
    
    def get_launch_power_dbm(self):
        if self.Plaunch_dbm is None:
            raise Exception("Launch power has not been calculated yet!")

        return self.Plaunch_dbm

    def forward(self, v):
        # Assumes that v is the input voltage (normalized to the [0, 1] range)
        z = self.pp_voltage * v + self.bias_voltage + self.voltage_insertion_loss

        # Calculate absorption from voltage
        alpha_db = self.spline_object.evaluate(z).squeeze()

        # Return transmitted field amplitude
        y = torch.sqrt(self.laser_power * torch.pow(10.0, -alpha_db / 10.0))

        # Calculate chirp
        chirp = self.linewidth_enhancement / 2 * torch.log(torch.square(y))

        # Log launch power
        self.Plaunch_dbm = 10.0 * torch.log10(torch.mean(torch.square(y)) / 1e-3)

        return y * torch.exp(1j * chirp)


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

    def get_thermal_noise_std(self):
        return self.thermal_noise_std

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


class DigitalToAnalogConverter(object):
    """
        DAC with bandwidth limitation modeled by a Bessel filter
    """
    def __init__(self, bwl_cutoff, dac_min_max, fs, bit_resolution=None, bessel_order=5,
                 out_power_dbfs=-6, dtype=torch.float64) -> None:
        # Set attributes of DAC
        self.dac_min_max = dac_min_max  # max-absolute value to be used in the digital signal
        self.power_dbfs = out_power_dbfs  # power after signal normalization to leave headroom for peaks
        self.bit_resolution = bit_resolution

        # Initialize bessel filter
        self.lpf = AllPassFilter()
        if bwl_cutoff is not None:
            self.lpf = BesselFilter(bessel_order=bessel_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)

    def set_bitres(self, new_bitres):
        assert isinstance(new_bitres, int) or new_bitres is None
        self.bit_resolution = new_bitres
    
    def get_sample_delay(self):
        return self.lpf.get_sample_delay()
    
    def get_lpf_filter(self):
        return self.lpf.get_filters()

    def forward(self, x):
        # Map digital signal to a voltage
        v = (x + (self.dac_min_max)) / (2 * self.dac_min_max)
        v = v / torch.sqrt(torch.mean(torch.square(v))) * 10 ** (self.power_dbfs / 20)  # normalize power

        # Run lpf
        v_lp = self.lpf.forward(v)
        
        return v_lp

    def eval(self, x):
        # Map digital signal to a voltage
        v = (x + (self.dac_min_max)) / (2 * self.dac_min_max)
        v = v / torch.sqrt(torch.mean(torch.square(v))) * 10 ** (self.power_dbfs / 20)  # normalize power

        if torch.any(v > 1.0) or torch.any(v < 0.0):
            print(f"WARNING: Over/underflow in DAC (min: {v.min()}, max: {v.max()})")

        if self.bit_resolution:
            v = quantize(v, self.bit_resolution)

        print(f"DAC: Power in digital domain: {10.0 * torch.log10(torch.mean(torch.square(v)))} [dBFS]")

        # Run lpf
        v_lp = self.lpf.forward_numpy(v)
        
        return v_lp


class AnalogToDigitalConverter(object):
    """
        ADC with bandwidth limitation modeled by a Bessel filter
    """
    def __init__(self, bwl_cutoff, fs, bit_resolution=None, bessel_order=5, dtype=torch.float64) -> None:
        # Set attributes
        self.bit_resolution = bit_resolution
        self.dtype = dtype

        # Initialize bessel filter
        self.lpf = AllPassFilter()
        if bwl_cutoff is not None:
            self.lpf = BesselFilter(bessel_order=bessel_order, cutoff_hz=bwl_cutoff, fs=fs, dtype=dtype)
    
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