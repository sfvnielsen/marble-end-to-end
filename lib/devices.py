"""
    Collection of devices used in transmission systems
"""

import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from scipy.interpolate import CubicSpline
from scipy.optimize import newton


class ElectroAbsorptionModulator(object):
    """
        Implementation of and electro absorption modulator (EAM) as used in

        E. M. Liang and J. M. Kahn,
        “Geometric Shaping for Distortion-Limited Intensity Modulation/Direct Detection Data Center Links,”
        IEEE Photonics Journal, vol. 15, no. 6, pp. 1–17, 2023, doi: 10.1109/JPHOT.2023.3335398.


        absorption transfer function derived from

        A. D. Gallant and J. C. Cartledge,
        “Characterization of the Dynamic Absorption of Electroabsorption Modulators With Application to OTDM Demultiplexing,”
        Journal of Lightwave Technology, vol. 26, no. 13, pp. 1835–1839, Jul. 2008, doi: 10.1109/JLT.2008.922190.

    """

    # Voltage-to-absorption curve derived from (Gallant and Cartledge, 2008) (Figure 3a, CIP static curve)
    # Read knee points and apply cubic spline fit - flip to supply monotonically increasing values to spline object
    #ABSORPTION_KNEE_POINTS_X = torch.flip(torch.Tensor([0.0, -0.5,  -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]), (0,))  # driving voltage
    #ABSORPTION_KNEE_POINTS_Y = torch.flip(torch.Tensor([0.0,  1.0,   2.5,  7.0, 20.0, 28.0, 28.0, 26.0, 24.0]), (0,))  # absorption in dB

    # Voltage-to-absorption curve directly from (Liang and Kahn, 2023)
    ABSORPTION_KNEE_POINTS_X = torch.flip(torch.Tensor([0.0, -0.5,  -1.0, -2.0, -3.0, -3.5,  -3.8]), (0,))  # driving voltage
    ABSORPTION_KNEE_POINTS_Y = torch.flip(torch.Tensor([0.0, 1.25,   2.5,  5.0,  9.5, 13.0,  12.5]), (0,))  # absorption in dB

    # Linear Voltage-to-absorption curve - used for debugging
    LINEAR_ABSORPTION_KNEE_POINTS_X = torch.flip(torch.Tensor([0.0, -0.5,  -1.0, -2.0, -3.0, -4.0]), (0,))  # driving voltage
    LINEAR_ABSORPTION_KNEE_POINTS_Y = torch.flip(torch.Tensor([0.0, 1.0,   2.0,  4.0,   6.0,  8.0]), (0,))  # absorption in dB


    def __init__(self, insertion_loss, pp_voltage, bias_voltage, laser_power, dac_min_max, linear_absorption=False):
        # Parse input arguments
        self.insertion_loss = insertion_loss
        self.pp_voltage = pp_voltage
        self.bias_voltage = bias_voltage
        self.laser_power = laser_power

        self.x_knee_points = self.ABSORPTION_KNEE_POINTS_X if not linear_absorption else self.LINEAR_ABSORPTION_KNEE_POINTS_X
        self.y_knee_points = self.ABSORPTION_KNEE_POINTS_Y if not linear_absorption else self.LINEAR_ABSORPTION_KNEE_POINTS_Y

        # TODO: Modulator chirp

        # Caculate the cubic spline coefficients based on the given knee-points
        spline_coefficients = natural_cubic_spline_coeffs(self.x_knee_points, self.y_knee_points[:, None])
        self.spline_object = NaturalCubicSpline(spline_coefficients)

        # Calculate voltage corresponding to given insertion loss
        cubicspline = CubicSpline(self.x_knee_points.numpy(), self.y_knee_points.numpy())
        self.voltage_insertion_loss = newton(lambda x: cubicspline(x) - self.insertion_loss, x0=-1.0)
        self.voltage_min = self.bias_voltage + self.pp_voltage / 2 + self.voltage_insertion_loss

        # Input to forward is assumed to be a digital signal - convert to voltages using dac_min_max
        # (typically based on constellation values)
        self.x_min, self.x_max = dac_min_max

    def forward(self, x):
        # Convert x to voltage (based on insertion loss and pp_voltage)
        z = (x - self.x_min) / (self.x_max - self.x_min)  # normalize
        v = self.pp_voltage * z + self.bias_voltage + self.voltage_insertion_loss

        # Calculate absorption from voltage
        alpha_db = self.spline_object.evaluate(v).squeeze()

        # Return transmitted field amplitude
        return torch.sqrt(self.laser_power * torch.pow(10.0, -alpha_db / 10.0))
