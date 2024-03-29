"""
    Module containing numpy/torch implementations of transmission systems (symbols -> symbols)
"""

import numpy.typing as npt
import numpy as np
import torch

from scipy.signal import bessel, lfilter, freqz, group_delay
from scipy.fft import fftshift
from commpy.filters import rrcosfilter

from .filtering import FIRfilter, BesselFilter, BrickWallFilter, AllPassFilter, filter_initialization
from .utility import find_max_variance_sample
from .devices import ElectroAbsorptionModulator

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

    def forward_batched(self, symbols_up: torch.TensorType, batch_size: int) -> torch.TensorType:
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

        if return_loss:
            return loss_per_batch

    def evaluate(self, symbols: npt.ArrayLike):
        # Upsample
        symbols_up = np.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols
        symbols_up = torch.from_numpy(symbols_up)
        # Run forward pass without gradient information - run batched version to not run into memory problems
        with torch.no_grad():
            rx_out = self.forward_batched(symbols_up, batch_size=self.eval_batch_size)

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
            __, g = rrcosfilter(tx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
             # Construct RRC filter
            __, g = rrcosfilter(rx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
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

    def forward_batched(self, symbols_up: torch.TensorType, batch_size: int):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_batched(symbols_up, batch_size)

        # Normalize (if self.normalization_after_tx is set, else norm_constant = 1.0)
        x = x / self.normalization_constant

        # Add white noise
        noise_std = self.calculate_noise_std(x)
        y = x + noise_std * torch.randn(x.shape)

        # Apply rx filter
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
                 dac_bwl_relative_cutoff, learn_tx: bool, learn_rx: bool,
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
            __, g = rrcosfilter(tx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
             # Construct RRC filter
            __, g = rrcosfilter(rx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
            g = g / np.linalg.norm(g)
            rx_filter_init = g

        self.rx_filter = FIRfilter(filter_weights=rx_filter_init, trainable=learn_rx, stride=self.sps)

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

        # Add white noise
        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp)
        y = x_lp + noise_std * torch.randn(x_lp.shape)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        # Apply rx filter
        rx_filter_out = self.rx_filter.forward(y_lp)

        # Power normalize and rescale to constellation
        rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def forward_batched(self, symbols_up: torch.TensorType, batch_size: int):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_batched(symbols_up, batch_size)

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
        rx_filter_out = self.rx_filter.forward_batched(y_lp, batch_size)

        # Power normalize and rescale to constellation
        rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

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
            __, g = rrcosfilter(tx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
             # Construct RRC filter
            __, g = rrcosfilter(rx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
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

    def forward_batched(self, symbols_up: torch.TensorType, batch_size: int):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_batched(symbols_up, batch_size)

        # Normalize
        x = x / self.normalization_constant

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward_batched(x, batch_size)

        # Apply non-linearity - FIR + non linear + FIR
        x_nl = self.isi_filter1.forward_batched(x_lp, batch_size)
        x_nl = self.non_linear_function(x_nl)
        x_nl = self.isi_filter2.forward_batched(x_nl, batch_size)

        # Add white noise
        noise_std = self.calculate_noise_std(x_nl)
        y = x_nl + noise_std * torch.randn(x_nl.shape)

        # Apply band-width limitation in the ADC
        y_lp = self.adc.forward_batched(y, batch_size)

        # Apply rx filter
        rx_filter_out = self.rx_filter.forward_batched(y_lp, batch_size)

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
                                                      photodiode
                                                          |
          <-  symbol decision <-  filtering <- adc <- noise (awgn)

    """
    def __init__(self, sps, noise_std, baud_rate, learning_rate, batch_size, constellation,
                 eam_insertion_loss_db, eam_voltage_pp, eam_laser_power, eam_voltage_bias,
                 learn_rx, learn_tx, rx_filter_length, tx_filter_length, square_law_photodiode,
                 eam_linear_absorption=False, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75, rrc_rolloff=0.5, 
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
            __, g = rrcosfilter(tx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
            g = g / np.linalg.norm(g)
            tx_filter_init = g

        self.pulse_shaper = FIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)

        # Define rx filter - downsample to 1 sps as part of convolution (stride)
        rx_filter_init = np.zeros((rx_filter_length,))
        if learn_rx and rx_filter_init_type != 'rrc':
            rx_filter_init = filter_initialization(rx_filter_init, rx_filter_init_type)
        else:
             # Construct RRC filter
            __, g = rrcosfilter(rx_filter_length, rrc_rolloff, self.sym_length, 1 / self.Ts)
            g = g[1::]  # delete first element to make filter odd length
            assert len(g) % 2 == 1  # we assume that pulse is always odd
            g = g / np.linalg.norm(g)
            rx_filter_init = g

        self.rx_filter = FIRfilter(filter_weights=rx_filter_init, trainable=learn_rx, stride=self.sps)

        # Define bandwidth limitation filters - low pass filter with cutoff relative to bw of RRC
        rrc_bw = 0.5 / (1 / baud_rate)

        # Digital-to-analog (DAC) converter
        self.dac = AllPassFilter()
        if dac_bwl_relative_cutoff is not None:
            self.dac = BesselFilter(bessel_order=5, cutoff_hz=rrc_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)

        # Analog-to-digial (ADC) converter
        self.adc = AllPassFilter()
        if adc_bwl_relative_cutoff is not None:
            self.adc = BesselFilter(bessel_order=5, cutoff_hz=rrc_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)

        # Define EAM
        xmax = np.sqrt(np.max(self.constellation)**2 / self.sps)  # assumes that constellation is symmetric around zero
        self.eam = ElectroAbsorptionModulator(insertion_loss=eam_insertion_loss_db,
                                            pp_voltage=eam_voltage_pp,
                                            bias_voltage=eam_voltage_bias,
                                            laser_power=eam_laser_power,
                                            dac_min_max=(-xmax, xmax),  # conversion from digital signal to voltage
                                            linear_absorption=eam_linear_absorption)

        # Define photodiode
        self.photodiode = lambda x: x  # linear photodiode
        if square_law_photodiode:
            self.photodiode = torch.square
        self.noise_std = noise_std
        self.Es = None  # initialize energy-per-symbol to None as it will be calculated on the fly during eval

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int(((self.pulse_shaper.filter_length + self.rx_filter.filter_length) // 2) / self.sps)

        # Calculate estimate of channel delay (in symbols)
        self.channel_delay = int(np.ceil((self.adc.get_sample_delay() + self.dac.get_sample_delay()) / sps))
        print(f"Channel delay is {self.channel_delay} [symbols]")

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

    def get_esn0_db(self):
        if self.Es is None:
            print('Warning! Evaluation was not run yet so EsN0 has been calculated yet.')
            return np.nan

        return (10.0 * np.log10(self.Es / self.noise_std**2)).item()

    def set_esn0_db(self, new_esn0_db):
        raise Exception(f"Cannot set EsN0 in this type of channel. Noise is given. Modify v_pp instead.")

    def set_energy_pr_symbol(self, x):
        # Calculate empirical symbol power (average over all the symbol periods)
        es = torch.mean(torch.sum(torch.square(torch.reshape(x - x.mean(), (-1, self.sps))), dim=1))
        # Converting average energy pr. symbol into base power (cf. https://wirelesspi.com/pulse-amplitude-modulation-pam/)
        self.Es = es * (3 / (len(self.constellation)**2 - 1))

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

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward(x)

        # Apply EAM
        x_eam = self.eam.forward(x_lp)

        # Photodiode - square law detection (if configured, else identity)
        x_pd = self.photodiode(x_eam)

        # Add white noise
        y = x_pd + self.noise_std * torch.randn(x_pd.shape)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        # Normalize
        y_norm = (y_lp - y_lp.mean()) / (y_lp.std())

        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        rx_filter_out = self.rx_filter.forward(y_norm)

        # Power normalize and rescale to constellation
        rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def forward_batched(self, symbols_up: torch.TensorType, batch_size: int):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward_batched(symbols_up, batch_size)

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward_batched(x, batch_size)

        # Apply EAM
        x_eam = self.eam.forward(x_lp)

        # Photodiode - square law detection (if configured, else identity)
        x_pd = self.photodiode(x_eam)

        # Add white noise
        self.set_energy_pr_symbol(x_pd)
        y = x_pd + self.noise_std * torch.randn(x_pd.shape)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward_batched(y, batch_size)

        # Normalize
        y_norm = (y_lp - y_lp.mean()) / (y_lp.std())

        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        rx_filter_out = self.rx_filter.forward_batched(y_norm, batch_size)

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


class PulseShapingIM(IntensityModulationChannel):
    """
        PulseShaping (learning Tx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, noise_std, baud_rate, learning_rate, batch_size, constellation,
                 eam_insertion_loss_db, eam_voltage_pp, eam_laser_power, eam_voltage_bias,
                 rx_filter_length, tx_filter_length, square_law_photodiode,
                 eam_linear_absorption=False,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75,
                 rrc_rolloff=0.5, use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, noise_std=noise_std, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         eam_insertion_loss_db=eam_insertion_loss_db, eam_voltage_pp=eam_voltage_pp,
                         eam_laser_power=eam_laser_power, eam_voltage_bias=eam_voltage_bias,
                         eam_linear_absorption=eam_linear_absorption,
                         learn_rx=False, learn_tx=True, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length, square_law_photodiode=square_law_photodiode,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)


class RxFilteringIM(IntensityModulationChannel):
    """
        RxFiltering (learning Rx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, noise_std, baud_rate, learning_rate, batch_size, constellation,
                 eam_insertion_loss_db, eam_voltage_pp, eam_laser_power, eam_voltage_bias,
                 rx_filter_length, tx_filter_length, square_law_photodiode,
                 eam_linear_absorption=False, 
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75,
                 rrc_rolloff=0.5, use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, noise_std=noise_std, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         eam_insertion_loss_db=eam_insertion_loss_db, eam_voltage_pp=eam_voltage_pp,
                         eam_laser_power=eam_laser_power, eam_voltage_bias=eam_voltage_bias,
                         eam_linear_absorption=eam_linear_absorption,
                         learn_rx=True, learn_tx=False, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length, square_law_photodiode=square_law_photodiode,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)


class JointTxRxIM(IntensityModulationChannel):
    """
        JointTxRx (learning both Tx andRx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, noise_std, baud_rate, learning_rate, batch_size, constellation,
                 eam_insertion_loss_db, eam_voltage_pp, eam_laser_power, eam_voltage_bias,
                 rx_filter_length, tx_filter_length, square_law_photodiode,
                 eam_linear_absorption=False,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_bwl_relative_cutoff=0.75, adc_bwl_relative_cutoff=0.75,
                 rrc_rolloff=0.5, use_1clr=False, eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, noise_std=noise_std, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         eam_insertion_loss_db=eam_insertion_loss_db, eam_voltage_pp=eam_voltage_pp,
                         eam_laser_power=eam_laser_power, eam_voltage_bias=eam_voltage_bias,
                         eam_linear_absorption=eam_linear_absorption,
                         learn_rx=True, learn_tx=True, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length, square_law_photodiode=square_law_photodiode,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff, adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         rrc_rolloff=rrc_rolloff, use_1clr=use_1clr, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval)
