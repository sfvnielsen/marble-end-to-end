"""
    Module containing numpy/torch implementations of transmission systems (symbols -> symbols)
"""

import numpy.typing as npt
import numpy as np
import torch

from torch.fft import fft, ifft, fftfreq
from scipy.signal import bessel, lfilter, freqz, group_delay
from scipy.fft import fftshift
from commpy.filters import rrcosfilter

from .filtering import FIRfilter, BesselFilter, BrickWallFilter, AllPassFilter, GaussianFqFilter, filter_initialization
from .utility import find_max_variance_sample, symbol_sync
from .devices import ElectroAbsorptionModulator, MyNonLinearEAM, Photodiode,\
                     IdealLinearModulator, DigitalToAnalogConverter, AnalogToDigitalConverter,\
                     MachZehnderModulator
from .channels import SingleModeFiber
from .equalization import LinearFeedForwardEqualiser

# TODO: Implement GPU support

class StandardTransmissionSystem(object):
    """
        Parent class for vanilla transmission system without any learnable parameters (evaluation only)
        Numpy based
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, normalize_after_tx=True) -> None:
        self.esn0_db = esn0_db
        self.baud_rate = baud_rate
        self.sps = sps  # samples pr symbol
        self.sym_length = 1 / self.baud_rate  # length of one symbol in seconds
        self.Ts = self.sym_length / self.sps  # effective sampling interval
        self.constellation = constellation
        self.normalize_after_tx = normalize_after_tx

    def set_esn0(self, new_esn0_db):
        self.esn0_db = new_esn0_db

    def optimize(self, symbols):
        print('This is an evaluation class. Skipping optimization step.')

    def initialize_optimizer(self):
        pass

    def get_esn0_db(self):
        return self.esn0_db

    def evaluate(self, symbols: npt.ArrayLike):
        raise NotImplementedError

    def calculate_noise_std(self, input_signal):
        """
            Calculate the standard deviation of AWGN with desired EsN0 based on an input signal
        """
        esn0 = (10 ** (self.esn0_db / 10.0))  # linear domain EsN0 ratio
        # Calculate empirical average energy pr. symbol of input signal
        n_syms = len(input_signal) // self.sps
        es = np.mean(np.sum(np.square(np.reshape(input_signal[0:n_syms * self.sps], (-1, self.sps))), axis=1))
        # Converting average energy pr. symbol into base power (cf. https://wirelesspi.com/pulse-amplitude-modulation-pam/)
        base_power_pam = es * (3 / (len(self.constellation)**2 - 1))
        n0 =  base_power_pam / esn0  # derive noise variance
        return np.sqrt(n0)


class MatchedFilterAWGN(StandardTransmissionSystem):
    """
        Special case of the BasicAWGN model with no learning. Rx and Tx set to matched filters.
    """
    def __init__(self, sps, snr_db, baud_rate, constellation, normalize_after_tx=False,
                 rrc_length_in_symbols=16, rrc_rolloff=0.5) -> None:
        super().__init__(sps=sps, esn0_db=snr_db, baud_rate=baud_rate,
                         constellation=constellation, normalize_after_tx=normalize_after_tx)

        # Construct RRC filter
        __, g = rrcosfilter(rrc_length_in_symbols * self.sps, rrc_rolloff, self.sym_length, 1 / self.Ts)
        g = g[1::]  # delete first element to make filter odd length
        assert len(g) % 2 == 1  # we assume that pulse is always odd
        g = g / np.linalg.norm(g)
        self.rrc_filter = g
        gg = np.convolve(self.rrc_filter, self.rrc_filter[::-1])
        self.sync_point = np.argmax(gg)
        self.pulse_energy = np.max(gg)

        # Calculate energy pr symbol and noise std for the AWGN channel
        self.normalization_constant = np.sqrt(np.average(np.square(self.constellation)) / self.sps) if self.normalize_after_tx else 1.0
        self.constellation_scale = np.sqrt(np.average(np.square(self.constellation)))

    def evaluate(self, symbols: npt.ArrayLike):
        # Up-sample symbols
        symbols_up = np.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols

        # Apply pulse shaper
        x = np.convolve(symbols_up, self.rrc_filter) / self.normalization_constant

        # Add white noise
        noise_std = self.calculate_noise_std(x)
        y = x + noise_std * np.random.randn(*x.shape)

        # Apply rx filter
        rx_filter_out = np.convolve(y, self.rrc_filter[::-1])

        # Sample selection
        rx_filter_out = rx_filter_out[self.sync_point:-self.sync_point:self.sps]

        # Rescale to constellation
        rx_filter_out = rx_filter_out / np.sqrt(np.mean(np.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def get_pulse_shaping_filter(self):
        return np.copy(self.rrc_filter)

    def get_rx_filter(self):
        return np.copy(self.rrc_filter[::-1])


class MatchedFilterAWGNwithBWL(StandardTransmissionSystem):
    """
        Special case of the BasicAWGNwithBWL model with no learning. Rx and Tx set to matched filters.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, adc_bwl_relative_cutoff,
                 dac_bwl_relative_cutoff, rrc_length_in_symbols=16, rrc_rolloff=0.5, normalize_after_tx=True) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         constellation=constellation, normalize_after_tx=normalize_after_tx)

        # Construct RRC filter
        __, g = rrcosfilter(rrc_length_in_symbols * self.sps, rrc_rolloff, self.sym_length, 1 / self.Ts)
        g = g[1::]  # delete first element to make filter odd length
        assert len(g) % 2 == 1  # we assume that pulse is always odd
        g = g / np.linalg.norm(g)
        self.rrc_filter = g
        gg = np.convolve(self.rrc_filter, self.rrc_filter[::-1])
        self.sync_point = np.argmax(gg)
        self.pulse_energy = np.max(gg)

        # Construct DAC and ADC low-pass filter - set to None if no cutof is specified
        # Define bandwidth limitation filters - low pass filter with cutoff relative to bw of RRC
        info_bw = 0.5 * baud_rate

        self.adc_bessel_b, self.adc_bessel_a = None, None
        if adc_bwl_relative_cutoff:
            self.adc_bessel_b, self.adc_bessel_a = bessel(5, info_bw * adc_bwl_relative_cutoff, fs=1 / self.Ts, norm='mag')

        self.dac_bessel_b, self.dac_bessel_a = None, None
        if dac_bwl_relative_cutoff:
            self.dac_bessel_b, self.dac_bessel_a = bessel(5, info_bw * dac_bwl_relative_cutoff, fs=1 / self.Ts, norm='mag')

        # Calculate energy pr symbol and noise std for the AWGN channel
        self.normalization_constant = np.sqrt(np.average(np.square(self.constellation)) / self.sps) if self.normalize_after_tx else 1.0
        self.constellation_scale = np.sqrt(np.average(np.square(self.constellation)))

    def optimize(self, symbols: npt.ArrayLike):
        # Do nothing when calling optimize
        pass

    def evaluate(self, symbols: npt.ArrayLike):
        # Up-sample symbols
        symbols_up = np.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols

        # Apply pulse shaper
        x = np.convolve(symbols_up, self.rrc_filter) / self.normalization_constant

        # Apply DAC low-pass filter (if specified)
        if self.dac_bessel_a is not None and self.dac_bessel_b is not None:
            x = lfilter(self.dac_bessel_b, self.dac_bessel_a, x)

        # Add white noise
        noise_std = self.calculate_noise_std(x)
        y = x + noise_std * np.random.randn(*x.shape)

        # Low-pass filter (simulate ADC) (if specified)
        if self.adc_bessel_a is not None and self.adc_bessel_b is not None:
            y = lfilter(self.adc_bessel_b, self.adc_bessel_a, y)

        # Apply rx filter
        rx_filter_out = np.convolve(y, self.rrc_filter[::-1])

        # Sample selection
        max_var_samp = find_max_variance_sample(rx_filter_out[self.sync_point:-self.sync_point], sps=self.sps)
        rx_filter_out = rx_filter_out[(self.sync_point + max_var_samp):(self.sync_point + max_var_samp + len(symbols) * self.sps):self.sps]

        # Rescale to constellation
        rx_filter_out = rx_filter_out / np.sqrt(np.mean(np.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def get_pulse_shaping_filter(self):
        return np.copy(self.rrc_filter)

    def get_rx_filter(self):
        return np.copy(self.rrc_filter[::-1])


class LearnableTransmissionSystem(object):
    """
        Parent class for end-to-end learning.
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(5e4)) -> None:
        self.esn0_db = esn0_db
        self.baud_rate = baud_rate
        self.learning_rate = learning_rate  # learning rate of optimizer
        self.sps = sps  # samples pr symbol
        self.sym_length = 1 / self.baud_rate  # length of one symbol in seconds
        self.Ts = self.sym_length / self.sps  # effective sampling interval
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size_in_syms * self.sps  # FIXME: Expose to all the classes
        self.batch_print_interval = print_interval / self.batch_size
        self.optimizer = None
        self.constellation = constellation
        self.use_1clr = use_1clr

    def print_system_info(self):
        # FIXME
        pass

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.get_parameters(), lr=self.learning_rate)

    def set_esn0_db(self, new_esn0_db):
        self.esn0_db = new_esn0_db

    def forward(self, symbols_up: torch.TensorType) -> torch.TensorType:
        raise NotImplementedError

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True) -> torch.TensorType:
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def zero_gradients(self):
        raise NotImplementedError

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        raise NotImplementedError

    def get_esn0_db(self):
        return self.esn0_db

    def calculate_noise_std(self, input_signal):
        """
            Calculate the standard deviation of AWGN with desired EsN0 based on an input signal
        """
        esn0 = (10 ** (self.esn0_db / 10.0))  # linear domain EsN0 ratio
        # Calculate empirical average energy pr. symbol of input signal
        es = torch.mean(torch.sum(torch.square(torch.reshape(input_signal - input_signal.mean(), (-1, self.sps))), dim=1))
        # Converting average energy pr. symbol into base power (cf. https://wirelesspi.com/pulse-amplitude-modulation-pam/)
        base_power_pam = es * (3 / (len(self.constellation)**2 - 1))
        n0 =  base_power_pam / esn0  # derive noise variance
        return torch.sqrt(n0)

    def update_model(self, loss):
        loss.backward()

        if not self.optimizer:
            raise Exception("Optimizer was not initialized. Please call the 'initialize_optimizer' method before proceeding to optimize.")

        # Gradient norm clipping
        for pgroup in self.optimizer.param_groups:
            # FIXME: Abusing param groups a bit here.
            # So far each param group corresponds to exatcly one parameter.
            torch.nn.utils.clip_grad_norm_(pgroup['params'], 1.0)  # clip all gradients to unit norm

        # Take gradient step.
        self.optimizer.step()

    def optimize(self, symbols: npt.ArrayLike, return_loss=False):
        symbols_up = np.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols

        if return_loss:
            loss_per_batch = np.empty((len(symbols) // self.batch_size, ), dtype=np.float64)

        # Create learning rate scheduler - OneCLR
        if self.use_1clr:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                               max_lr=10 * self.learning_rate,
                                                               steps_per_epoch=1,
                                                               epochs=len(symbols) // self.batch_size)

        for b in range(len(symbols) // self.batch_size):
            # Zero gradients
            self.zero_gradients()

            # Slice out batch and create tensors
            this_a_up = symbols_up[b * self.batch_size * self.sps:(b * self.batch_size * self.sps + self.batch_size * self.sps)]
            target = torch.from_numpy(symbols[b * self.batch_size:(b * self.batch_size + self.batch_size)])
            tx_syms_up = torch.from_numpy(this_a_up)

            # Run upsampled symbols through system forward model - return symbols at Rx
            rx_out = self.forward(tx_syms_up)

            # Calculate loss
            loss = self.calculate_loss(target, rx_out)

            # Update model using backpropagation
            self.update_model(loss)

            this_lr = self.optimizer.param_groups[-1]['lr']
            if self.use_1clr:
                lr_scheduler.step()
                this_lr = lr_scheduler.get_last_lr()[0]

            if b % self.batch_print_interval == 0:
                print(f"Batch {b} (# symbols {b * self.batch_size:.2e}) - Loss: {loss.item():.3f} - LR: {this_lr:.2e}")

            if return_loss:
                loss_per_batch[b] = loss.item()

            if torch.isnan(loss):
                print("Detected loss to be nan. Terminate training...")
                break

        if return_loss:
            return loss_per_batch

    def evaluate(self, symbols: npt.ArrayLike, **eval_config):
        # Upsample
        symbols_up = np.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols
        symbols_up = torch.from_numpy(symbols_up)
        # Run forward pass without gradient information - run batched version to not run into memory problems
        with torch.no_grad():
            rx_out = self._eval(symbols_up, batch_size=self.eval_batch_size, **eval_config)

        return rx_out.detach().cpu().numpy()


class BasicAWGN(LearnableTransmissionSystem):
    """
        Basic additive white Gaussian noise system with pulse-shaping by RRC
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation, learn_tx: bool, learn_rx: bool, print_interval=int(5e4),
                 normalize_after_tx=True, tx_filter_length=45, rx_filter_length=45,
                 tx_filter_init_type='dirac', rx_filter_init_type='dirac', rrc_rolloff=0.5, use_1clr=False) -> None:
        super().__init__(sps=sps,
                         esn0_db=esn0_db,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         print_interval=print_interval,
                         use_1clr=use_1clr)

        # Define pulse shaper
        tx_filter_init = np.zeros((tx_filter_length,))
        if learn_tx and tx_filter_init_type != 'rrc':
            tx_filter_init = filter_initialization(tx_filter_init, tx_filter_init_type)
        else:
            # Construct RRC filter
            assert tx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(tx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
            # Construct RRC filter
            assert rx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(rx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            rx_filter_init = g

        self.rx_filter = FIRfilter(filter_weights=rx_filter_init, trainable=learn_rx, stride=self.sps)

        # Set the post-tx-normalization-property of the class
        self.normalize_after_tx = normalize_after_tx
        self.normalization_constant = np.sqrt(np.average(np.square(self.constellation)) / self.sps) if self.normalize_after_tx else 1.0

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int((self.pulse_shaper.filter_length + self.rx_filter.filter_length) / self.sps)

    def get_parameters(self):
        params_to_return = []
        params_to_return.append({"params": self.pulse_shaper.parameters()})
        params_to_return.append({"params": self.rx_filter.parameters()})
        return params_to_return

    def zero_gradients(self):
        self.pulse_shaper.zero_grad()
        self.rx_filter.zero_grad()

    def forward(self, symbols_up: torch.TensorType):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(symbols_up)

        # Normalize (if self.normalization_after_tx is set, else norm_constant = 1.0)
        x = x / self.normalization_constant

        # Add white noise based on desired EsN0
        with torch.no_grad():
            noise_std = self.calculate_noise_std(x)

        y = x + noise_std * torch.randn(x.shape)

        # Apply rx filter
        rx_filter_out = self.rx_filter.forward(y)

        # Rescale to constellation (if self.normalization_after_tx is set)
        if self.normalize_after_tx:
            rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_batched(symbols_up, batch_size)

        # Normalize (if self.normalization_after_tx is set, else norm_constant = 1.0)
        x = x / self.normalization_constant

        # Add white noise
        noise_std = self.calculate_noise_std(x)
        y = x + noise_std * torch.randn(x.shape)

        # Apply rx filter
        if not decimate:
            self.rx_filter.set_stride(1)  # output all samples from rx_filter
        rx_filter_out = self.rx_filter.forward_batched(y, batch_size)

        # Rescale to constellation (if self.normalization_after_tx is set)
        if self.normalize_after_tx:
            rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        return torch.mean(torch.square(tx_syms[self.discard_per_batch:-self.discard_per_batch] - rx_syms[self.discard_per_batch:-self.discard_per_batch]))

    def update_model(self, loss):
        super().update_model(loss)
        # Projected gradient - Normalize filters
        self.pulse_shaper.normalize_filter()
        self.rx_filter.normalize_filter()

    def get_pulse_shaping_filter(self):
        return self.pulse_shaper.get_filter()

    def get_rx_filter(self):
        return self.rx_filter.get_filter()


class PulseShapingAWGN(BasicAWGN):
    """
        Special case of the BasicAWGN model learning the Tx filter (Rx filter set to RRC)
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate, tx_filter_length,
                 rx_filter_length, print_interval=int(50000), rrc_rolloff=0.5,
                 normalize_after_tx=True, filter_init_type='dirac', use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=True, learn_rx=False,
                         tx_filter_length=tx_filter_length,
                         rx_filter_length=rx_filter_length,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         normalize_after_tx=normalize_after_tx,
                         tx_filter_init_type=filter_init_type,
                         rx_filter_init_type='rrc',
                         use_1clr=use_1clr)


class RxFilteringAWGN(BasicAWGN):
    """
        Special case of the BasicAWGN model learning the Rx filter (Tx filter set to RRC)
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate, rx_filter_length,
                 tx_filter_length, print_interval=int(50000), rrc_rolloff=0.5,
                 normalize_after_tx=True, filter_init_type='dirac', use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=False, learn_rx=True,
                         tx_filter_length=tx_filter_length,  # filter length controlled by RRC
                         rx_filter_length=rx_filter_length,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         normalize_after_tx=normalize_after_tx,
                         rx_filter_init_type=filter_init_type,
                         tx_filter_init_type='rrc',
                         use_1clr=use_1clr)


class BasicAWGNwithBWL(LearnableTransmissionSystem):
    """
        Basic additive white Gaussian noise system with pulse-shaping by RRC and bandwidth limitation
        Bandwidth limitation parameter is defined relative to the bandwidth of the RRC pulse.
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation,
                 tx_filter_length: int, rx_filter_length: int, adc_bwl_relative_cutoff,
                 dac_bwl_relative_cutoff, learn_tx: bool, learn_rx: bool, equaliser_config: dict | None = None,
                 tx_filter_init_type='dirac', rx_filter_init_type='dirac',
                 print_interval=int(5e4), use_brickwall=False, use_1clr=False,
                 rrc_rolloff=0.5) -> None:
        super().__init__(sps=sps,
                         esn0_db=esn0_db,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         print_interval=print_interval,
                         use_1clr=use_1clr)

        # Define pulse shaper
        tx_filter_init = np.zeros((tx_filter_length,))
        if learn_tx and tx_filter_init_type != 'rrc':
            tx_filter_init = filter_initialization(tx_filter_init, tx_filter_init_type)
        else:
            # Construct RRC filter
            assert tx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(tx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
            # Construct RRC filter
            assert rx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(rx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            rx_filter_init = g

        # Check if we equalization has been specified - if we do, do not downsample after rx filter
        self.use_eq = bool(equaliser_config)
        self.rx_filter = FIRfilter(filter_weights=rx_filter_init, trainable=learn_rx, stride=1 if self.use_eq else self.sps)

        # Define equaliser object
        self.equaliser = AllPassFilter()
        if self.use_eq:
            self.equaliser = LinearFeedForwardEqualiser(samples_per_symbol=self.sps, dtype=torch.float64,  # FIXME: !
                                                        **equaliser_config)

        # Define bandwidth limitation filters - low pass filter with cutoff relative to bandwidth of baseband
        info_bw = 0.5 * baud_rate

        # Digital-to-analog (DAC) converter
        self.dac = AllPassFilter()
        if dac_bwl_relative_cutoff is not None:
            if use_brickwall:
                self.dac = BrickWallFilter(filter_length=512, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                self.dac = BesselFilter(bessel_order=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)

        # Analog-to-digial (ADC) converter
        self.adc = AllPassFilter()

        if adc_bwl_relative_cutoff is not None:
            if use_brickwall:
                self.adc = BrickWallFilter(filter_length=512, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                self.adc = BesselFilter(bessel_order=5, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)

        self.normalization_constant = np.sqrt(np.average(np.square(self.constellation)) / self.sps)

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int((self.pulse_shaper.filter_length + self.rx_filter.filter_length) / self.sps)

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

        # Total symbol delay introduced by the two LPFs
        self.channel_delay = int(np.ceil((self.adc.get_sample_delay() + self.dac.get_sample_delay())/self.sps))
        print(f"Channel delay is {self.channel_delay} [symbols]")

    def get_parameters(self):
        params_to_return = []
        params_to_return.append({"params": self.pulse_shaper.parameters()})
        params_to_return.append({"params": self.rx_filter.parameters()})
        if self.use_eq:
            params_to_return += self.equaliser.get_parameters()
        return params_to_return

    def zero_gradients(self):
        self.pulse_shaper.zero_grad()
        self.rx_filter.zero_grad()
        self.equaliser.zero_grad()

    def forward(self, symbols_up: torch.TensorType):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(symbols_up)

        # Normalize
        x = x / self.normalization_constant

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward(x)

        # Add white noise
        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp)
        y = x_lp + noise_std * torch.randn(x_lp.shape)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        # Apply rx filter
        rx_filter_out = self.rx_filter.forward(y_lp)

        # Apply equaliser
        eq_out = self.equaliser.forward(rx_filter_out)

        # Power normalize and rescale to constellation
        eq_out = eq_out / torch.sqrt(torch.mean(torch.square(eq_out))) * self.constellation_scale

        return eq_out

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_numpy(symbols_up)

        # Normalize
        x = x / self.normalization_constant

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward_batched(x, batch_size)

        # Add white noise
        noise_std = self.calculate_noise_std(x_lp)
        y = x_lp + noise_std * torch.randn(x_lp.shape)

        # Apply band-width limitation in the ADC
        y_lp = self.adc.forward_batched(y, batch_size)

        # Apply rx filter
        if not decimate:
            self.rx_filter.set_stride(1)  # output all samples from rx_filter
        rx_filter_out = self.rx_filter.forward_numpy(y_lp)

        # Apply equaliser
        if not decimate and self.use_eq:
            self.equaliser.set_stride(1)  # output all samples from equaliser
        rx_eq_out = self.equaliser.forward_batched(rx_filter_out, batch_size)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        return torch.mean(torch.square(tx_syms[self.discard_per_batch:-self.discard_per_batch] - torch.roll(rx_syms, -self.channel_delay)[self.discard_per_batch:-self.discard_per_batch]))

    def update_model(self, loss):
        super().update_model(loss)
        # Projected gradient - Normalize filters
        self.pulse_shaper.normalize_filter()
        self.rx_filter.normalize_filter()

    def get_pulse_shaping_filter(self):
        return self.pulse_shaper.get_filter()

    def get_rx_filter(self):
        return self.rx_filter.get_filter()


class PulseShapingAWGNwithBWL(BasicAWGNwithBWL):
    """
        PulseShaper in bandwidth limited AWGN channel
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 tx_filter_length, rx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 filter_init_type='dirac', print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=True, learn_rx=False,
                         tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length,
                         tx_filter_init_type=filter_init_type,
                         rx_filter_init_type='rrc',
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)


class RxFilteringAWGNwithBWL(BasicAWGNwithBWL):
    """
       Bandwidth limited AWGN channel with learnable Rx filter.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 filter_init_type='dirac', print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=False, learn_rx=True,
                         rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=filter_init_type,
                         tx_filter_init_type='rrc',
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)


class JointTxRxAWGNwithBWL(BasicAWGNwithBWL):
    """
       Bandwidth limited AWGN channel with learnable Tx and Rx filter.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=True, learn_rx=True,
                         tx_filter_length=tx_filter_length,
                         tx_filter_init_type=tx_filter_init_type,
                         rx_filter_length=rx_filter_length,
                         rx_filter_init_type=rx_filter_init_type,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)


class LinearFFEAWGNwithBWL(BasicAWGNwithBWL):
    """
       Bandwidth limited AWGN channel with fixed Tx and Rx filters.
       Adaptive FFE equaliser to combat ISI.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 ffe_n_taps, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         equaliser_config={'n_taps': ffe_n_taps},
                         learn_tx=False, learn_rx=False,
                         tx_filter_length=tx_filter_length,
                         tx_filter_init_type=tx_filter_init_type,
                         rx_filter_length=rx_filter_length,
                         rx_filter_init_type=rx_filter_init_type,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)

    def get_equaliser_filter(self):
        return self.equaliser.filter.get_filter()


class BasicAWGNwithBWLandWDM(BasicAWGNwithBWL):
    """
        Bandwidth limited AWGN channel with wavelength division multiplexing (WDM)
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation,
                 tx_filter_length: int, rx_filter_length: int, adc_bwl_relative_cutoff,
                 dac_bwl_relative_cutoff, learn_tx: bool, learn_rx: bool,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 equaliser_config: dict | None = None, tx_filter_init_type='dirac',
                 rx_filter_init_type='dirac', print_interval=int(50000), torch_seed=0,
                 use_brickwall=False, use_1clr=False, rrc_rolloff=0.5) -> None:
        super().__init__(sps, esn0_db, baud_rate, learning_rate, batch_size, constellation,
                         tx_filter_length, rx_filter_length, adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff, learn_tx, learn_rx, equaliser_config,
                         tx_filter_init_type, rx_filter_init_type, print_interval,
                         use_brickwall, use_1clr, rrc_rolloff)
        self.wdm_n_channels = 3  # always three channels during training
        self.wdm_channel_spacing_hz = wdm_channel_spacing_hz
        self.torch_seed = torch_seed  # used for generating interferer channels

        # Create Gauss filter object for channel selection on Rx side
        self.channel_selection_filter = GaussianFqFilter(filter_cutoff_hz=(0.5 * self.baud_rate) * wdm_channel_selection_rel_cutoff,
                                                         order=5,
                                                         Fs=1/self.Ts)

    def forward(self, symbols_up: torch.TensorType):
        rng_gen = torch.random.manual_seed(self.torch_seed)
        symbols_up_chan = torch.zeros((symbols_up.shape[0], self.wdm_n_channels), dtype=symbols_up.dtype)
        symbols_up_chan[::, self.wdm_n_channels // 2] = symbols_up
        channels_to_fill = [i for i in range(self.wdm_n_channels) if i != self.wdm_n_channels // 2]  # all except middle
        for cf in channels_to_fill:
            symbols_up_chan[0::self.sps, cf] = symbols_up[::self.sps][torch.randperm(symbols_up.shape[0] // self.sps, generator=rng_gen)]

        # Prepare WDM channel
        tx_wdm = torch.zeros((symbols_up_chan.shape[0], ), dtype=torch.complex128)
        channel_fq_grid = torch.arange(-np.floor(self.wdm_n_channels / 2), np.floor(self.wdm_n_channels / 2) + 1, 1) * self.wdm_channel_spacing_hz

        for c in range(self.wdm_n_channels):
            x = self.pulse_shaper.forward(symbols_up_chan[:, c])

            x = x / self.normalization_constant

            # Apply bandwidth limitation in the DAC
            x_lp = self.dac.forward(x)

            tx_wdm += x_lp * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[c] * self.Ts) * torch.arange(0, x.shape[0], dtype=torch.float64))

        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp)  # just calculate the noise_std based on the last channel
        y = tx_wdm + noise_std * torch.randn(x_lp.shape)

        # Low-pass filter to select middle channel
        y_chan = torch.real(self.channel_selection_filter.forward(y))

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y_chan)

        # Apply rx filter - applies stride inside filter (outputs sps = 1, if no equaliser)
        rx_filter_out = self.rx_filter.forward(y_lp)

        # Apply equaliser
        rx_eq_out = self.equaliser.forward(rx_filter_out)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out
    
    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True):
        rng_gen = torch.random.manual_seed(self.torch_seed)
        symbols_up_chan = torch.zeros((symbols_up.shape[0], self.wdm_n_channels), dtype=symbols_up.dtype)
        symbols_up_chan[::, self.wdm_n_channels // 2] = symbols_up
        channels_to_fill = [i for i in range(self.wdm_n_channels) if i != self.wdm_n_channels // 2]  # all except middle
        for cf in channels_to_fill:
            symbols_up_chan[0::self.sps, cf] = symbols_up[::self.sps][torch.randperm(symbols_up.shape[0] // self.sps, generator=rng_gen)]

        # Prepare WDM channel
        tx_wdm = torch.zeros((symbols_up_chan.shape[0], ), dtype=torch.complex128)
        channel_fq_grid = torch.arange(-np.floor(self.wdm_n_channels / 2), np.floor(self.wdm_n_channels / 2) + 1, 1) * self.wdm_channel_spacing_hz

        for c in range(self.wdm_n_channels):
            x = self.pulse_shaper.forward_numpy(symbols_up_chan[:, c])

            x = x / self.normalization_constant

            # Apply bandwidth limitation in the DAC
            x_lp = self.dac.forward_batched(x, batch_size=batch_size)

            tx_wdm += x_lp * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[c] * self.Ts) * torch.arange(0, x.shape[0], dtype=torch.float64))

        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp)  # just calculate the noise_std based on the last channel
        y = tx_wdm + noise_std * torch.randn(x_lp.shape)

        # Low-pass filter to select middle channel
        y_chan = torch.real(self.channel_selection_filter.forward_numpy(y))

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward_batched(y_chan, batch_size=batch_size)

        # Apply rx filter - applies stride inside filter (outputs sps = 1, if no equalsier)
        if not decimate:
            self.rx_filter.set_stride(1)  # output all samples from rx_filter
        rx_filter_out = self.rx_filter.forward_numpy(y_lp)

        # Apply equaliser
        if not decimate and self.use_eq:
            self.equaliser.set_stride(1)  # output all samples from equaliser
        rx_eq_out = self.equaliser.forward_batched(rx_filter_out, batch_size)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

class PulseShapingAWGNwithBWLandWDM(BasicAWGNwithBWLandWDM):
    """
        PulseShaper in bandwidth limited AWGN channel with WDM.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 tx_filter_length, rx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 filter_init_type='dirac', print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=True, learn_rx=False,
                         tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length,
                         tx_filter_init_type=filter_init_type,
                         rx_filter_init_type='rrc',
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)


class RxFilteringAWGNwithBWLandWDM(BasicAWGNwithBWLandWDM):
    """
       Bandwidth limited AWGN channel with learnable Rx filter and WDM.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 filter_init_type='dirac', print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=False, learn_rx=True,
                         rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=filter_init_type,
                         tx_filter_init_type='rrc',
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)


class JointTxRxAWGNwithBWLandWDM(BasicAWGNwithBWLandWDM):
    """
       Bandwidth limited AWGN channel (WDM) with learnable Tx and Rx filter.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=True, learn_rx=True,
                         tx_filter_length=tx_filter_length,
                         tx_filter_init_type=tx_filter_init_type,
                         rx_filter_length=rx_filter_length,
                         rx_filter_init_type=rx_filter_init_type,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)


class LinearFFEAWGNwithBWLandWDM(BasicAWGNwithBWLandWDM):
    """
       Bandwidth limited AWGN channel with fixed Tx and Rx filters and WDM.
       Adaptive FFE equaliser to combat ISI.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 ffe_n_taps, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5,
                 use_brickwall=False, use_1clr=False) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         equaliser_config={'n_taps': ffe_n_taps},
                         learn_tx=False, learn_rx=False,
                         tx_filter_length=tx_filter_length,
                         tx_filter_init_type=tx_filter_init_type,
                         rx_filter_length=rx_filter_length,
                         rx_filter_init_type=rx_filter_init_type,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         use_brickwall=use_brickwall,
                         use_1clr=use_1clr)

    def get_equaliser_filter(self):
        return self.equaliser.filter.get_filter()


class NonLinearISIChannel(LearnableTransmissionSystem):
    """
        General non-linear channel with additive Gaussian noise and bandwidth limitation
        Non-linearity is comprised of a FIR filter cascaded with a non-linear transformation,
        and finally another FIR filter.
        Non-linearity is parameterized by a cubic function
            f(x) = a_0 x + a_1 x ** 2 + a_2 ** 3
        The coefficients a = [a_0, a_1, a_2] are specified as the input argument 'non_linear_coefficients'
        Bandwidth limitation parameter is defined relative to the bandwidth of the RRC pulse.
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation, adc_bwl_relative_cutoff,
                 dac_bwl_relative_cutoff, learn_tx: bool, learn_rx: bool, tx_filter_length: int, rx_filter_length: int,
                 non_linear_coefficients=(0.95, 0.04, 0.01), isi_filter1=np.array([0.2, -0.1, 0.9, 0.3]),
                 isi_filter2=np.array([0.2, 0.9, 0.3]), tx_filter_init_type='rrc', rx_filter_init_type='rrc',
                 print_interval=int(5e4), use_brickwall=False, use_1clr=False,
                 rrc_rolloff=0.5) -> None:
        super().__init__(sps=sps,
                         esn0_db=esn0_db,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         print_interval=print_interval,
                         use_1clr=use_1clr)

        # Define pulse shaper
        tx_filter_init = np.zeros((tx_filter_length,))
        if learn_tx and tx_filter_init_type != 'rrc':
            tx_filter_init = filter_initialization(tx_filter_init, tx_filter_init_type)
        else:
            # Construct RRC filter
            assert tx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(tx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
            # Construct RRC filter
            assert rx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(rx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            rx_filter_init = g

        self.rx_filter = FIRfilter(filter_weights=rx_filter_init, trainable=learn_rx, stride=self.sps)

        # Define bandwidth limitation filters - low pass filter with cutoff relative to bw of RRC
        info_bw = 0.5 * baud_rate

        # Digital-to-analog (DAC) converter
        self.dac = AllPassFilter()
        if dac_bwl_relative_cutoff is not None:
            if use_brickwall:
                self.dac = BrickWallFilter(filter_length=512, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                self.dac = BesselFilter(bessel_order=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)

        # Analog-to-digial (ADC) converter
        self.adc = AllPassFilter()

        if adc_bwl_relative_cutoff is not None:
            if use_brickwall:
                self.adc = BrickWallFilter(filter_length=512, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                self.adc = BesselFilter(bessel_order=5, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)

        # Add parameters for non-linear channel
        # ISI transfer functions are assumed to be in "symbol"-domain, i.e. how much two neighbouring symbols interfere
        h1_isi_zeropadded = np.zeros(self.sps * (len(isi_filter1) - 1) + 1)
        h1_isi_zeropadded[::self.sps] = isi_filter1
        h1_isi_zeropadded = h1_isi_zeropadded / np.linalg.norm(h1_isi_zeropadded)
        self.isi_filter1 = FIRfilter(filter_weights=h1_isi_zeropadded)

        self.non_linear_function = lambda x: non_linear_coefficients[0] * x + non_linear_coefficients[1] * x**2 + non_linear_coefficients[2] * x**3

        h2_isi_zeropadded = np.zeros(self.sps * (len(isi_filter2) - 1) + 1)
        h2_isi_zeropadded[::self.sps] = isi_filter2
        h2_isi_zeropadded = h2_isi_zeropadded / np.linalg.norm(h2_isi_zeropadded)
        self.isi_filter2 = FIRfilter(filter_weights=h2_isi_zeropadded)

        # Estimate the sample delay introduced by the ISI filters
        f, gd = group_delay((h1_isi_zeropadded, 1), fs=1/self.Ts)
        isi1_delay = np.average(gd[np.where(f < info_bw)])
        f, gd = group_delay((h2_isi_zeropadded, 1), fs=1/self.Ts)
        isi2_delay = np.average(gd[np.where(f < info_bw)])
        isi_delay = isi1_delay + isi2_delay

        # Calculate normalization constant after Tx filter
        self.normalization_constant = np.sqrt(np.average(np.square(self.constellation)) / self.sps)

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int(((self.pulse_shaper.filter_length + self.rx_filter.filter_length) // 2) / self.sps)

        # Calculate estimate of channel delay (in symbols)
        self.channel_delay = int(np.ceil((isi_delay + self.adc.get_sample_delay() + self.dac.get_sample_delay()) / sps))
        print(f"Channel delay is {self.channel_delay} [symbols] (ISI contributed with {int(np.ceil(isi_delay / sps))})")

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

    def get_parameters(self):
        params_to_return = []
        params_to_return.append({"params": self.pulse_shaper.parameters()})
        params_to_return.append({"params": self.rx_filter.parameters()})
        return params_to_return

    def zero_gradients(self):
        self.pulse_shaper.zero_grad()
        self.rx_filter.zero_grad()

    def forward(self, symbols_up: torch.TensorType):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(symbols_up)

        # Normalize
        x = x / self.normalization_constant

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward(x)

        # Apply non-linearity - FIR + non linear + FIR
        x_nl = self.isi_filter1.forward(x_lp)
        x_nl = self.non_linear_function(x_nl)
        x_nl = self.isi_filter2.forward(x_nl)

        # Add white noise
        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_nl)
        y = x_nl + noise_std * torch.randn(x_nl.shape)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        rx_filter_out = self.rx_filter.forward(y_lp)

        # Power normalize and rescale to constellation
        rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_numpy(symbols_up)

        # Normalize
        x = x / self.normalization_constant

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward_batched(x, batch_size)

        # Apply non-linearity - FIR + non linear + FIR
        x_nl = self.isi_filter1.forward_numpy(x_lp)
        x_nl = self.non_linear_function(x_nl)
        x_nl = self.isi_filter2.forward_numpy(x_nl)

        # Add white noise
        noise_std = self.calculate_noise_std(x_nl)
        y = x_nl + noise_std * torch.randn(x_nl.shape)

        # Apply band-width limitation in the ADC
        y_lp = self.adc.forward_batched(y, batch_size)

        # Apply rx filter
        if not decimate:
            self.rx_filter.set_stride(1)  # output all samples from rx_filter
        rx_filter_out = self.rx_filter.forward_numpy(y_lp, batch_size)

        # Power normalize and rescale to constellation
        rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        # NB! Rx sequence is coarsely aligned to the tx-symbol sequence based on a apriori known channel delay.
        return torch.mean(torch.square(tx_syms[self.discard_per_batch:-self.discard_per_batch] - torch.roll(rx_syms, -self.channel_delay)[self.discard_per_batch:-self.discard_per_batch]))

    def update_model(self, loss):
        super().update_model(loss)
        # Projected gradient - Normalize filters
        self.pulse_shaper.normalize_filter()
        self.rx_filter.normalize_filter()

    def get_pulse_shaping_filter(self):
        return self.pulse_shaper.get_filter()

    def get_rx_filter(self):
        return self.rx_filter.get_filter()

    def get_total_isi(self):
        return np.convolve(self.isi_filter1.get_filter(), self.isi_filter2.get_filter())


class PulseShapingNonLinearISIChannel(NonLinearISIChannel):
    """
        Learning the Tx filter in the non-linear isi channel
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size,
                 constellation, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 tx_filter_length, rx_filter_length,
                 non_linear_coefficients=(0.95, 0.04, 0.01),
                 isi_filter1=np.array([0.2, -0.1, 0.9, 0.3]), isi_filter2=np.array([0.2, 0.9, 0.3]),
                 filter_init_type='rrc', print_interval=int(50000), use_brickwall=False,
                 use_1clr=False, rrc_rolloff=0.5) -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learn_tx=True, learn_rx=False,
                         non_linear_coefficients=non_linear_coefficients,
                         isi_filter1=isi_filter1, isi_filter2=isi_filter2,
                         tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length,
                         tx_filter_init_type=filter_init_type, rx_filter_init_type='rrc',
                         print_interval=print_interval, use_brickwall=use_brickwall, use_1clr=use_1clr,
                         rrc_rolloff=rrc_rolloff)


class RxFilteringNonLinearISIChannel(NonLinearISIChannel):
    """
        Learning the Rx filter in the non-linear isi channel
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size,
                 constellation, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 rx_filter_length, tx_filter_length, non_linear_coefficients=(0.95, 0.04, 0.01),
                 isi_filter1=np.array([0.2, -0.1, 0.9, 0.3]), isi_filter2=np.array([0.2, 0.9, 0.3]),
                 filter_init_type='rrc', print_interval=int(50000),
                 use_brickwall=False, use_1clr=False, rrc_rolloff=0.5) -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learn_tx=False, learn_rx=True,
                         non_linear_coefficients=non_linear_coefficients,
                         isi_filter1=isi_filter1, isi_filter2=isi_filter2,
                         tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length,
                         tx_filter_init_type='rrc', rx_filter_init_type=filter_init_type,
                         print_interval=print_interval, use_brickwall=use_brickwall, use_1clr=use_1clr,
                         rrc_rolloff=rrc_rolloff)


class JointTxRxNonLinearISIChannel(NonLinearISIChannel):
    """
        Learning both Tx and Rx filters in the non-linear isi channel
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size,
                 constellation, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 tx_filter_length, rx_filter_length,
                 non_linear_coefficients=(0.95, 0.04, 0.01),
                 isi_filter1=np.array([0.2, -0.1, 0.9, 0.3]), isi_filter2=np.array([0.2, 0.9, 0.3]),
                 tx_filter_init_type='rrc',
                 rx_filter_init_type='rrc', print_interval=int(50000), use_brickwall=False,
                 use_1clr=False, rrc_rolloff=0.5) -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learn_tx=True, learn_rx=True,
                         non_linear_coefficients=non_linear_coefficients,
                         isi_filter1=isi_filter1, isi_filter2=isi_filter2,
                         tx_filter_length=tx_filter_length, rx_filter_length=rx_filter_length,
                         tx_filter_init_type=tx_filter_init_type, rx_filter_init_type=rx_filter_init_type,
                         print_interval=print_interval, use_brickwall=use_brickwall, use_1clr=use_1clr,
                         rrc_rolloff=rrc_rolloff)


class IntensityModulationChannel(LearnableTransmissionSystem):
    """
        Intensity modulation/direct detection (IM/DD) system inspired by

        E. M. Liang and J. M. Kahn,
        “Geometric Shaping for Distortion-Limited Intensity Modulation/Direct Detection Data Center Links,”
        IEEE Photonics Journal, vol. 15, no. 6, pp. 1–17, 2023, doi: 10.1109/JPHOT.2023.3335398.


        The system implements an electro absorption modulator (EAM), based on
        absorption curves derived from the above reference.

        System has the following structure

        symbols -> upsampling -> pulse shaping -> dac -> eam
                                                          |
                                                        channel (SMF)
                                                          |
          <-  symbol decision <-  filtering <- adc <- photodiode

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 learn_rx, learn_tx, rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias,
                 modulator_type='eam', equaliser_config: dict | None = None,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 dac_bitres=None, adc_bitres=None, dac_minmax_norm: float | str = 'auto',
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, esn0_db=np.nan, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, use_1clr=use_1clr,
                         eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval)

        # Define pulse shaper
        tx_filter_init = np.zeros((tx_filter_length,))
        if learn_tx and tx_filter_init_type != 'rrc':
            tx_filter_init = filter_initialization(tx_filter_init, tx_filter_init_type)
        else:
            # Construct RRC filter
            assert tx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(tx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
            # Construct RRC filter
            assert rx_filter_length % 2 == 1  # we assume that pulse is always odd
            __, g = rrcosfilter(rx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            rx_filter_init = g

        # Check if we equalization has been specified - if we do, do not downsample after rx filter
        self.use_eq = bool(equaliser_config)
        self.rx_filter = FIRfilter(filter_weights=rx_filter_init, trainable=learn_rx, stride=1 if self.use_eq else self.sps)

        # Define equaliser object
        self.equaliser = AllPassFilter()
        if self.use_eq:
            self.equaliser = LinearFeedForwardEqualiser(samples_per_symbol=self.sps, dtype=torch.float64,  # FIXME: !
                                                        **equaliser_config)

        # Define bandwidth limitation filters - low pass filter with cutoff relative to bw of baseband
        info_bw = 0.5 * baud_rate

        # Estimate the min-max value of an RRC filter empirically and apply that as normalization in the DAC
        dac_normalizer = dac_minmax_norm
        if isinstance(dac_minmax_norm, float) or isinstance(dac_minmax_norm, int):
            pass
        elif isinstance(dac_minmax_norm, str) and dac_minmax_norm == 'auto':
            __, g = rrcosfilter(tx_filter_length + 1, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            g = g / np.linalg.norm(g)
            randomgen = np.random.default_rng(0)
            a = randomgen.choice(constellation, size=(int(1e5),), replace=True)
            aup = np.zeros((len(a) * self.sps, ))
            aup[::self.sps] = a
            x = np.convolve(aup, g)
            dac_normalizer = 2 * np.abs(np.max(x))
        elif isinstance(dac_minmax_norm, str) and dac_minmax_norm == 'minmax':
            dac_normalizer = dac_minmax_norm
        else:
            raise Exception(f"Unknown DAC normalization scheme: '{dac_minmax_norm}'")

        # Digital-to-analog (DAC) converter
        self.dac = DigitalToAnalogConverter(bias_voltage=dac_voltage_bias, peak_to_peak_voltage=dac_voltage_pp,
                                            peak_to_peak_constellation=dac_normalizer,
                                            bwl_cutoff=None if dac_bwl_relative_cutoff is None else info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts,
                                            bessel_order=5, bit_resolution=dac_bitres)

        # Analog-to-digial (ADC) converter
        self.adc = AnalogToDigitalConverter(bwl_cutoff=None if adc_bwl_relative_cutoff is None else info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts,
                                            bessel_order=5, bit_resolution=adc_bitres)

        # Define modulator
        if modulator_type == 'ideal':
            self.modulator = IdealLinearModulator(laser_power_dbm=modulator_config['laser_power_dbm'])
        elif modulator_type == 'mzm':
            self.modulator = MachZehnderModulator(**modulator_config)
        elif modulator_type == 'eam':
            self.modulator = ElectroAbsorptionModulator(**modulator_config)
        elif modulator_type == 'nonlin_eam':
            self.modulator = MyNonLinearEAM(**modulator_config)
        else:
            raise Exception(f"Unknown modulator type '{modulator_type}'. Valid options are: 'ideal', 'eam' or 'nonlin_eam'")

        # Define channel - single mode fiber with chromatic dispersion
        self.channel = SingleModeFiber(Fs=1/self.Ts, **smf_config)

        # Define photodiode
        self.photodiode = Photodiode(bandwidth=info_bw * adc_bwl_relative_cutoff if adc_bwl_relative_cutoff is not None else info_bw,
                                     Fs=1/self.Ts, sps=self.sps, **photodiode_config)
        self.Es = None  # initialize energy-per-symbol to None as it will be calculated on the fly during eval

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int(((self.pulse_shaper.filter_length + self.rx_filter.filter_length) // 2) / self.sps)

        # Calculate estimate of channel delay (in symbols)
        self.channel_delay = int(np.ceil((self.adc.get_sample_delay() + self.dac.get_sample_delay()) / sps))
        print(f"Channel delay is {self.channel_delay} [symbols]")

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

    def get_esn0_db(self):
        # Get theoretical estimate of EsN0
        # Assumes that the system is thermal-noise limited
        if self.Es is None:
            print('Warning! Evaluation was not run yet so EsN0 has been calculated yet.')
            return np.nan

        return (10.0 * np.log10(self.Es / self.photodiode.get_thermal_noise_std()**2)).item()

    def get_launch_power_dbm(self):
        return self.modulator.get_launch_power_dbm().item()

    def get_received_power_dbm(self):
        return self.photodiode.get_received_power_dbm().item()

    def set_esn0_db(self, new_esn0_db):
        raise Exception(f"Cannot set EsN0 in this type of channel. Noise is given. Modify v_pp or eam laser power instead.")

    def set_energy_pr_symbol(self, es):
        # Converting average energy pr. symbol into base power (cf. https://wirelesspi.com/pulse-amplitude-modulation-pam/)
        self.Es = es * (3 / (len(self.constellation)**2 - 1))

    def get_parameters(self):
        params_to_return = []
        params_to_return.append({"params": self.pulse_shaper.parameters()})
        params_to_return.append({"params": self.rx_filter.parameters()})
        if self.use_eq:
            params_to_return += self.equaliser.get_parameters()
        return params_to_return

    def zero_gradients(self):
        self.pulse_shaper.zero_grad()
        self.rx_filter.zero_grad()
        if self.use_eq:
            self.equaliser.zero_grad()

    def forward(self, symbols_up: torch.TensorType):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(symbols_up)

        # Apply bandwidth limitation in the DAC
        v = self.dac.forward(x)

        # Apply EAM
        x_eam = self.modulator.forward(v)

        # Apply channel model
        x_chan = self.channel.forward(x_eam)

        # Photodiode - square law detection - adds noise inside (thermal and shot noise)
        y = self.photodiode.forward(x_chan)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        # Normalize
        y_norm = (y_lp - y_lp.mean()) / (y_lp.std())

        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        # (if equaliser is not specified)
        rx_filter_out = self.rx_filter.forward(y_norm)

        # Apply equaliser
        rx_eq_out = self.equaliser.forward(rx_filter_out)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True, **eval_config):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_numpy(symbols_up)

        # Apply bandwidth limitation in the DAC
        self.dac.set_bitres(eval_config.get('dac_bitres', None))
        v = self.dac.eval(x)
        print(f"DAC: Voltage min {v.min()}, Voltage max {v.max()}")

        # Apply EAM
        x_eam = self.modulator.forward(v)
        print(f"Modulator: Laser power {10.0 * np.log10(self.modulator.laser_power / 1e-3) } [dBm]")
        print(f"Modulator: Power at output {10.0 * np.log10(np.average(np.square(np.absolute(x_eam.detach().numpy()))) / 1e-3)} [dBm]")

        # Apply channel model
        x_chan = self.channel.forward(x_eam)

        # Photodiode - square law detection - adds noise inside (thermal and shot noise)
        y = self.photodiode.forward(x_chan)
        self.set_energy_pr_symbol(self.photodiode.Es)
        print(f"Photodiode: Received power {self.photodiode.get_received_power_dbm()} [dBm]")
        print(f"Photodiode: Received Es: {10.0 * np.log10(self.photodiode.Es / 1e-3)} [dBm]")

        # Apply bandwidth limitation in the ADC
        self.adc.set_bitres(eval_config.get('adc_bitres', None))
        y_lp = self.adc.eval(y)

        # Normalize
        y_norm = (y_lp - y_lp.mean()) / (y_lp.std())

        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        if not decimate:
            self.rx_filter.set_stride(1)  # output all samples from rx_filter
        rx_filter_out = self.rx_filter.forward_numpy(y_norm)

        # Apply equaliser
        if not decimate and self.use_eq:
            self.equaliser.set_stride(1)  # output all samples from equaliser
        rx_eq_out = self.equaliser.forward_batched(rx_filter_out, batch_size)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        # NB! Rx sequence is coarsely aligned to the tx-symbol sequence based on a apriori known channel delay.
        return torch.mean(torch.square(tx_syms[self.discard_per_batch:-self.discard_per_batch] - torch.roll(rx_syms, -self.channel_delay)[self.discard_per_batch:-self.discard_per_batch]))

    def update_model(self, loss):
        super().update_model(loss)
        # Projected gradient - Normalize filters
        self.pulse_shaper.normalize_filter()
        self.rx_filter.normalize_filter()

    def get_pulse_shaping_filter(self):
        return self.pulse_shaper.get_filter()

    def get_rx_filter(self):
        return self.rx_filter.get_filter()


class PulseShapingIM(IntensityModulationChannel):
    """
        PulseShaping (learning Tx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias,
                 modulator_type='eam', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 dac_bitres=None, dac_minmax_norm: float | str = 'auto', adc_bitres=None,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         learn_rx=False, learn_tx=True, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         dac_minmax_norm=dac_minmax_norm,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         modulator_type=modulator_type,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bitres=dac_bitres, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)


class RxFilteringIM(IntensityModulationChannel):
    """
        RxFiltering (learning Rx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 dac_bitres=None, adc_bitres=None,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         dac_minmax_norm=dac_minmax_norm,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         modulator_type=modulator_type, learn_rx=True, learn_tx=False,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bitres=dac_bitres, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)

class JointTxRxIM(IntensityModulationChannel):
    """
        JointTxRx (learning both Tx andRx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, dac_minmax_norm: float | str = 'auto',
                 modulator_type=False, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 dac_bitres=None, adc_bitres=None,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         modulator_type=modulator_type,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         dac_minmax_norm=dac_minmax_norm,
                         learn_rx=True, learn_tx=True, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bitres=dac_bitres, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)


class LinearFFEIM(IntensityModulationChannel):
    """
        RRC + Matched filter + Linear FFE in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, dac_minmax_norm: float | str = 'auto',
                 modulator_type=False, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 dac_bitres=None, adc_bitres=None,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         dac_minmax_norm=dac_minmax_norm,
                         modulator_type=modulator_type, equaliser_config={'n_taps': ffe_n_taps},
                         learn_rx=False, learn_tx=False, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bitres=dac_bitres, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)

    def get_equaliser_filter(self):
        return self.equaliser.filter.get_filter()


class IntensityModulationChannelwithWDM(IntensityModulationChannel):
    """
        Intensity modulation/direct detection (IM/DD) system inspired by

        E. M. Liang and J. M. Kahn,
        “Geometric Shaping for Distortion-Limited Intensity Modulation/Direct Detection Data Center Links,”
        IEEE Photonics Journal, vol. 15, no. 6, pp. 1–17, 2023, doi: 10.1109/JPHOT.2023.3335398.

        with wavelength division multiplexing (WDM).

        The system implements an electro absorption modulator (EAM), based on
        absorption curves derived from the above reference.

        System has the following structure during evaluation

        tx : symbols -> upsampling -> pulse shaping -> dac -> eam

        [tx] x n_channels -> WDM shift and add ->  singe mode fiber
                                                          |
                                                    channel selection
                                                      (filtering)
                                                          |
          <-  symbol decision <- filtering <- adc <- photodiode

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 learn_rx, learn_tx, rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, wdm_channel_spacing_hz,
                 wdm_channel_selection_rel_cutoff, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', equaliser_config: dict | None = None,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, use_1clr=use_1clr,
                         learn_rx=learn_rx, learn_tx=learn_tx, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         dac_minmax_norm=dac_minmax_norm,
                         modulator_type=modulator_type, equaliser_config=equaliser_config,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, dac_bitres=None, adc_bitres=None,
                         eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval)

        self.wdm_n_channels = 3  # always three channels during training
        self.wdm_channel_spacing_hz = wdm_channel_spacing_hz
        self.torch_seed = torch_seed  # used for generating interferer channels

        # Create Gauss filter object for channel selection on Rx side
        self.channel_selection_filter = GaussianFqFilter(filter_cutoff_hz=(0.5 * self.baud_rate) * wdm_channel_selection_rel_cutoff,
                                                         order=5,
                                                         Fs=1/self.Ts)


    def forward(self, symbols_up: torch.TensorType):
        rng_gen = torch.random.manual_seed(self.torch_seed)
        symbols_up_chan = torch.zeros((symbols_up.shape[0], self.wdm_n_channels), dtype=symbols_up.dtype)
        symbols_up_chan[::, self.wdm_n_channels // 2] = symbols_up
        channels_to_fill = [i for i in range(self.wdm_n_channels) if i != self.wdm_n_channels // 2]  # all except middle
        for cf in channels_to_fill:
            symbols_up_chan[0::self.sps, cf] = symbols_up[::self.sps][torch.randperm(symbols_up.shape[0] // self.sps, generator=rng_gen)]

        # Prepare WDM channel
        tx_wdm = torch.zeros((symbols_up_chan.shape[0], ), dtype=torch.complex128)
        channel_fq_grid = torch.arange(-np.floor(self.wdm_n_channels / 2), np.floor(self.wdm_n_channels / 2) + 1, 1) * self.wdm_channel_spacing_hz

        for c in range(self.wdm_n_channels):
            x = self.pulse_shaper.forward(symbols_up_chan[:, c])

            # Apply bandwidth limitation in the DAC
            v = self.dac.forward(x)

            # Apply EAM
            x_eam = self.modulator.forward(v)

            tx_wdm += x_eam * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[c] * self.Ts) * torch.arange(0, x.shape[0], dtype=torch.float64))

        # Apply SMF
        x_smf = self.channel.forward(tx_wdm)

        # Low-pass filter to select middle channel
        x_chan = self.channel_selection_filter.forward(x_smf)

        # Photodiode
        y = self.photodiode.forward(x_chan)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        # Normalize
        y_norm = (y_lp - y_lp.mean()) / (y_lp.std())

        # Apply rx filter - applies stride inside filter (outputs sps = 1, if no equalsier)
        rx_filter_out = self.rx_filter.forward(y_norm)

        # Apply equaliser
        rx_eq_out = self.equaliser.forward(rx_filter_out)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    def eval_tx(self, symbols_up: torch.TensorType, n_channels: int, channel_spacing_hz: float,
                batch_size: int, dac_bitres: int | None = None, torch_seed: int = 0):
        # Generate interferer symbols - randomly permute the input sequence
        # Channel of interest will be the "middle" chanel (idx = n_channels // 2)
        rng_gen = torch.random.manual_seed(torch_seed)
        symbols_up_chan = torch.zeros((symbols_up.shape[0], n_channels), dtype=symbols_up.dtype)
        symbols_up_chan[::, n_channels // 2] = symbols_up
        channels_to_fill = [i for i in range(n_channels) if i != n_channels // 2]  # all except middle
        for cf in channels_to_fill:
            symbols_up_chan[0::self.sps, cf] = symbols_up[::self.sps][torch.randperm(symbols_up.shape[0] // self.sps, generator=rng_gen)]

        # Prepare WDM channel
        tx_wdm = torch.zeros((symbols_up_chan.shape[0], ), dtype=torch.complex128)
        channel_fq_grid = torch.arange(-np.floor(n_channels / 2), np.floor(n_channels / 2) + 1, 1) * channel_spacing_hz

        # Set DAC bit resolution
        self.dac.set_bitres(dac_bitres)

        print(f"Channel spacing: {channel_spacing_hz / 1e9} GHz")
        print(f"Channel grid: {channel_fq_grid / 1e9} GHz")

        for c in range(n_channels):
            x = self.pulse_shaper.forward_numpy(symbols_up_chan[:, c])

            # Apply bandwidth limitation in the DAC
            v = self.dac.eval(x)

            # Apply EAM
            x_eam = self.modulator.forward(v)
            print(f"EAM (channel {c}): Power at output {10.0 * np.log10(np.average(np.square(np.absolute(x_eam.detach().numpy()))) / 1e-3)} [dBm]")

            tx_wdm += x_eam * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[c] * self.Ts) * torch.arange(0, x.shape[0], dtype=torch.float64))

        return tx_wdm

    def eval_rx(self, x_chan: torch.TensorType, decimate: bool, batch_size: int, adc_bitres: int | None):
        # Apply photodiode
        y = self.photodiode.forward(x_chan)
        self.set_energy_pr_symbol(self.photodiode.Es)
        print(f"Photodiode: Received power {self.photodiode.get_received_power_dbm()} [dBm]")

        # Apply bandwidth limitation in the ADC
        self.adc.set_bitres(adc_bitres)
        y_lp = self.adc.eval(y)

        # Normalize
        y_norm = (y_lp - y_lp.mean()) / (y_lp.std())

        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        if not decimate:
            self.rx_filter.set_stride(1)  # output all samples from rx_filter
        rx_filter_out = self.rx_filter.forward_numpy(y_norm)

        # Apply equaliser
        if not decimate and self.use_eq:
            self.equaliser.set_stride(1)  # output all samples from equaliser
        rx_eq_out = self.equaliser.forward_batched(rx_filter_out, batch_size)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    # FIXME: Replace this with the acutal evaluate call.
    def _eval(self, symbols_up: torch.TensorType, batch_size: int, **eval_config):
        # Fetch config
        decimate = eval_config.get('decimate', True)
        channel_spacing_hz = eval_config.get('channel_spacing_hz', self.wdm_channel_spacing_hz)
        n_channels = eval_config.get('n_channels', self.wdm_n_channels)
        torch_seed = eval_config.get('seed', self.torch_seed)
        assert (n_channels + 1) % 2 == 0

        # Apply Tx (including generating interferer symbols and WDM signal)
        tx_wdm = self.eval_tx(symbols_up, n_channels, channel_spacing_hz, batch_size,
                              dac_bitres=eval_config.get('dac_bitres', None), torch_seed=torch_seed)

        # Apply SMF
        x_smf = self.channel.forward(tx_wdm)

        # Channel selection - filter out everything except the "middle" channel
        print(f"Power before channel selection: {10.0 * torch.log10(torch.mean(torch.square(torch.abs(x_smf))) / 1e-3)} [dBm]")
        x_chan = self.channel_selection_filter.forward_numpy(x_smf)

        # Apply Rx
        rx = self.eval_rx(x_chan, decimate, batch_size, adc_bitres=eval_config.get('adc_bitres', None))

        return rx


class PulseShapingIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Pulse-shaper is learned

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, wdm_channel_spacing_hz,
                 wdm_channel_selection_rel_cutoff,
                 modulator_type=False, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, dac_minmax_norm: float | str = 'auto',
                 adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, use_1clr=use_1clr,
                         learn_rx=False, learn_tx=True, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         modulator_type=modulator_type, equaliser_config=None,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_minmax_norm=dac_minmax_norm,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed)


class RxFilteringIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Rx-filter is learned

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, wdm_channel_spacing_hz,
                 wdm_channel_selection_rel_cutoff,
                 modulator_type='eam', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, dac_minmax_norm: float | str = 'auto',
                 adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, use_1clr=use_1clr,
                         learn_rx=True, learn_tx=False, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         modulator_type=modulator_type, equaliser_config=None,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, dac_minmax_norm=dac_minmax_norm,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed)


class JointTxRxIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Pulse-shaper and rx-filter is learned

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, wdm_channel_spacing_hz,
                 wdm_channel_selection_rel_cutoff,
                 modulator_type='eam', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, dac_minmax_norm: float | str = 'auto',
                 adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, use_1clr=use_1clr,
                         learn_rx=True, learn_tx=True, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         modulator_type=modulator_type, equaliser_config=None,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, dac_minmax_norm=dac_minmax_norm,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed)


class LinearFFEIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Pulse-shaper and rx-filter are fixed to RRC and linear equaliser is run afterwards

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps: int,
                 smf_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_voltage_pp, dac_voltage_bias, wdm_channel_spacing_hz,
                 wdm_channel_selection_rel_cutoff,
                 modulator_type='eam', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, dac_minmax_norm: float | str = 'auto',
                 adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5,
                 use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, use_1clr=use_1clr,
                         learn_rx=False, learn_tx=False, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         smf_config=smf_config, photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_voltage_pp=dac_voltage_pp, dac_voltage_bias=dac_voltage_bias,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         modulator_type=modulator_type, equaliser_config={'n_taps': ffe_n_taps},
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, dac_minmax_norm=dac_minmax_norm,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed)
