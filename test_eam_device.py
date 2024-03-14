"""
    Simple test and plot of the EAM
"""

import torch
import matplotlib.pyplot as plt

from lib.devices import ElectroAbsorptionModulator


if __name__ == "__main__":
    # Simulation parameters
    v_pp = 3.0  # peak-to-peak voltage
    insertion_loss_db = 0.0
    laser_power = 1.0
    bias = -2.0

    # Create the EAM object
    eam = ElectroAbsorptionModulator(insertion_loss=insertion_loss_db,
                                     pp_voltage=v_pp,
                                     laser_power=laser_power,
                                     bias_voltage=bias,
                                     dac_min_max=(-1, 1),  # normally specified in terms of constellation, but here we use a sine
                                     linear_absorption=False)

    # Evaluate for a given signal
    t = torch.linspace(0.0, 10.0, 10000)
    f = 3.0
    x = torch.sin(2 * torch.pi * f * t)

    # Apply EAM
    x_modulated = eam.forward(x)

    # Plot stuff
    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(x[0:1000])
    axs[1].plot(x_modulated[0:1000])

    # Plot voltage-to-absorption function - compare with (Liang and Kahn)
    v = torch.linspace(-4.0, 0.0, 1000)
    alpha_db = eam.spline_object.evaluate(v)
    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(v, alpha_db)
    axs[0].plot(eam.x_knee_points, eam.y_knee_points, 'ro')
    axs[0].axvline(eam.voltage_min, color='k', linestyle='--')
    axs[0].axvline(eam.voltage_min - eam.pp_voltage, color='k', linestyle='--')
    axs[0].axvline(eam.bias_voltage, color='r', linestyle='--')
    axs[0].set_xlabel('Voltage')
    axs[0].set_ylabel('Absorption [dB]')
    axs[0].invert_xaxis()
    axs[0].invert_yaxis()
    axs[0].grid()

    x = torch.linspace(-1.0, 1.0, 1000)
    axs[1].plot(x, eam.forward(x))
    axs[1].set_xlabel('Input signal [au]')
    axs[1].set_ylabel('Output amplitude')

    plt.show()