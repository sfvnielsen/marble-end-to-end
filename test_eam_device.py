"""
    Simple test and plot of the EAM
"""

import torch
import matplotlib.pyplot as plt

from lib.devices import ElectroAbsorptionModulator


if __name__ == "__main__":
    # Simulation parameters
    v_pp = 2.0  # peak-to-peak voltage
    insertion_loss_db = 1.0
    laser_power = 1.0
    bias = -2.0

    # Create the EAM object
    eam = ElectroAbsorptionModulator(insertion_loss=insertion_loss_db,
                                     pp_voltage=v_pp,
                                     laser_power=laser_power,
                                     bias_voltage=bias)

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
    fig, ax = plt.subplots()
    ax.plot(v, alpha_db)
    ax.plot(eam.ABSORPTION_KNEE_POINTS_X, eam.ABSORPTION_KNEE_POINTS_Y, 'ro')
    ax.axvline(eam.voltage_min, color='k', linestyle='--')
    ax.axvline(eam.voltage_min - eam.pp_voltage, color='k', linestyle='--')
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Absorption [dB]')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid()

    plt.show()