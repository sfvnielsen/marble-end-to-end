"""
    Simple test and plot of the EAM
    Test predistortion
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from commpy.filters import rrcosfilter

from lib.devices import ElectroAbsorptionModulator, DigitalToAnalogConverter, MyNonLinearEAM


if __name__ == "__main__":
    # Simulation parameters
    n_symbols = int(1e5)
    baud_rate = int(100e9)
    sps = 4
    seed = 12345
    pam_constellation = np.array([-3, -1, 1, 3])
    bessel_order = 5
    bessel_filter_cutoff_hz = (baud_rate * 0.5)
    Ts = 1 / (baud_rate * sps)

    v_pp = 3.0  # peak-to-peak voltage
    insertion_loss_db = 0.0
    laser_power = 0.0
    bias = -2.0

    # Create the EAM object
    eam = MyNonLinearEAM(laser_power_dbm=laser_power,
                                     linear_absorption=False)

    # Evaluate for a given signal
    random_obj = np.random.default_rng(1535345)
    x = np.zeros((n_symbols * sps))
    x[::sps] = random_obj.choice(pam_constellation, size=(n_symbols,))

    __, g = rrcosfilter(1024, 0.1, 1/baud_rate, 1/Ts)
    sync_point = np.argmax(np.convolve(g, g))
    x = np.convolve(x, g)
    
    # Convert to voltage
    dac = DigitalToAnalogConverter(bessel_filter_cutoff_hz, peak_to_peak_voltage=v_pp,
                                   bias_voltage=bias,
                                   fs=1/Ts, out_power_dbfs=-12)
    v = dac.eval(torch.from_numpy(x))
    vmin, vmax = v.min(), v.max()

    fig, ax = plt.subplots()
    ax.hist(v, bins=100)
    ax.set_title('Voltage distribution')

    # Apply EAM
    x_modulated = eam.forward(v)

    # Square law detection
    x_pd = np.square(np.absolute(x_modulated))

    # Normalize and apply RRC
    y = (x_pd - x_pd.mean()) / x_pd.std()
    rx = np.convolve(y, g)[sync_point:-sync_point:sps]

    # Plot stuff
    fig, ax = plt.subplots()
    ax.hist(rx, bins=100)

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(x[1000:2000])
    axs[1].plot(x_modulated[1000:2000])

    # Plot voltage-to-absorption function - compare with (Liang and Kahn)
    v = torch.linspace(-4.0, 0.0, 1000)
    alpha_db = eam.spline_object.evaluate(v)
    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(v, alpha_db)
    axs[0].plot(eam.x_knee_points, eam.y_knee_points, 'ro')
    axs[0].axvline(vmin, color='k', linestyle='--')
    axs[0].axvline(vmax, color='k', linestyle='--')
    axs[0].axvline(dac.v_bias, color='r', linestyle='--')
    axs[0].set_xlabel('Voltage')
    axs[0].set_ylabel('Absorption [dB]')
    axs[0].invert_xaxis()
    axs[0].invert_yaxis()
    axs[0].grid()

    x = torch.linspace(vmin, vmax, 1000)
    axs[1].plot(x, torch.absolute(eam.forward(x)))
    axs[1].set_xlabel('Voltage')
    axs[1].set_ylabel('Output amplitude')

    plt.show()
