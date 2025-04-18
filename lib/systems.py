"""
    Module containing numpy/torch implementations of transmission systems (symbols -> symbols)
"""

import numpy.typing as npt
import numpy as np
import torch

from commpy.filters import rrcosfilter
from copy import deepcopy
from torch.fft import fft, ifft, fftfreq
from scipy.signal import bessel, lfilter, freqz, group_delay
from scipy.fft import fftshift

from .filtering import FIRfilter, BesselFilter, BrickWallFilter, AllPassFilter, GaussianFqFilter,\
                       MultiChannelFIRfilter, MultiChannelBesselFilter, LowPassFIR, filter_initialization
from .utility import find_max_variance_sample, symbol_sync, permute_symbols
from .devices import ElectroAbsorptionModulator, MyNonLinearEAM, Photodiode,\
                     IdealLinearModulator, DigitalToAnalogConverter, AnalogToDigitalConverter,\
                     MachZehnderModulator
from .channels import SingleModeFiber, SplitStepFourierFiber, SurrogateChannel
from .equalization import LinearFeedForwardEqualiser, VolterraEqualizer

# TODO: Implement GPU support
# TODO: Be in control of dtypes.

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
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation, learn_tx, learn_rx,
                 lr_schedule='expdecay', eval_batch_size_in_syms=1000, print_interval=int(5e4),
                 tx_optimizer_params: dict | None = None, tx_multi_channel=False) -> None:
        self.esn0_db = esn0_db
        self.baud_rate = baud_rate
        self.sps = sps  # samples pr symbol
        self.sym_length = 1 / self.baud_rate  # length of one symbol in seconds
        self.Ts = self.sym_length / self.sps  # effective sampling interval
        self.batch_size = batch_size
        self.learning_rate = learning_rate  # learning rate used in the main Adam optmizer (default used for Rx)
        self.tx_learning_rate = None  # in case a separate tx optimizer is used, this is the learning rate.
        self.lr_schedule = lr_schedule
        self.eval_batch_size = eval_batch_size_in_syms * self.sps  # FIXME: Expose to all the classes
        self.batch_print_interval = print_interval / self.batch_size
        self.constellation = constellation
        self.learn_rx = learn_rx
        self.learn_tx = learn_tx
        self.tx_multi_channel = tx_multi_channel
        self.optimizer = None
        self.tx_optimize = None

        # Select optimization framework to use for the Tx
        self.optimize_method_funcp = self._optimize_backprop_funcp  # default is backprop
        if tx_optimizer_params:
            tx_optimizer_params_local = deepcopy(tx_optimizer_params)
            self.tx_optimizer_type = tx_optimizer_params_local.pop('type', 'backprop')
            if self.tx_optimizer_type.lower() == 'backprop':
                # Standard "backprop" uses the same
                # Assumes that we can differentiate through the channel
                self.use_gradient_norm_clipping = tx_optimizer_params_local.pop('gradient_norm_clipping', True)
            elif self.tx_optimizer_type.lower() == 'surrogate':
                # Differentiable surrogate channel (DSC) (cf. Niu et al. 2022, JLT, DOI: 10.1109/JLT.2022.3148270)
                self.optimize_method_funcp = self._optimize_surrogate_funcp

                # Alternating optimization - chunk sizes
                self.surrogate_chunk_size_pct = tx_optimizer_params_local.pop('chunk_size_pct', 0.1)

                # Frequency domain loss - penalized spectrum
                self.surrogate_fq_loss = tx_optimizer_params_local.pop('surrogate_fq_loss', False)

                # Tx optimizer - if learning rate is set to None, Tx and Rx loss is added together for combined optimization
                self.tx_learning_rate = tx_optimizer_params_local.pop('tx_learning_rate', None)  # learning rate surrogate optimizer
                self.tx_lr_schedule = tx_optimizer_params_local.pop('tx_lr_schedule', 'expdecay')
                self.use_gradient_norm_clipping = tx_optimizer_params_local.pop('tx_gradient_norm_clipping', True)

                # Surrgate optimizer
                self.surrogate_learning_rate = tx_optimizer_params_local.pop('surrogate_learning_rate', self.learning_rate)  # learning rate surrogate optimizer
                self.surrogate_lr_schedule = tx_optimizer_params_local.pop('surrogate_lr_schedule', 'expdecay')
                self.surrogate_channel = SurrogateChannel(multi_channel=tx_multi_channel, **tx_optimizer_params_local['surrogate_channel_kwargs'])

            elif self.tx_optimizer_type.lower() == 'cma-es':
                # Update with covariance matrix adaptation (use pycma package?)
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown optimizer type: '{self.tx_optimizer_type}'")
        else:
            # If no tx optimizer params are specified, assume vanilla backprop.
            self.tx_optimizer_type = 'backprop'
            self.use_gradient_norm_clipping = True

    def initialize_optimizer(self):
        params = self.get_rx_parameters()
        if self.tx_optimizer_type.lower() == 'backprop' or not self.tx_learning_rate:
            params += self.get_tx_parameters()
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        if self.tx_optimizer_type.lower() == 'surrogate': 
            self.surrogate_optimizer = torch.optim.Adam([{"params": self.surrogate_channel.parameters()}],
                                                        lr=self.surrogate_learning_rate)

            if self.tx_learning_rate:
                self.tx_optimizer = torch.optim.Adam(self.get_tx_parameters(),
                                                     lr=self.tx_learning_rate)

    def set_esn0_db(self, new_esn0_db):
        self.esn0_db = new_esn0_db

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True) -> torch.TensorType:
        raise NotImplementedError

    def get_tx_parameters(self):
        raise NotImplementedError

    def get_rx_parameters(self):
        raise NotImplementedError

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        raise NotImplementedError

    def calculate_surrogate_loss(self, y_pred: torch.TensorType, y_true: torch.TensorType):
        discard_samples = self.surrogate_channel.get_samples_discard()
        td_loss = torch.mean(torch.square(torch.subtract(y_pred[discard_samples:-discard_samples],
                                                         y_true[discard_samples:-discard_samples])))
        fq_loss = 0.0
        if self.surrogate_fq_loss:
            # TODO: Is it neccesarry with the dB conversion?
            fq_loss = torch.mean(torch.square(torch.subtract(20.0 * torch.log10(torch.absolute(fft(y_pred)) + 1e-16),
                                                             20.0 * torch.log10(torch.absolute(fft(y_true))+ 1e-16))))
        return td_loss + 1 / len(y_true) * fq_loss

    def get_esn0_db(self):
        return self.esn0_db

    def forward_tx(self, x_syms_up: torch.TensorType) -> torch.TensorType:
        raise NotImplementedError

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
        raise NotImplementedError

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        raise NotImplementedError

    # full forward pass
    def forward(self, symbols_up: torch.TensorType) -> torch.TensorType:
        tx  = self.forward_tx(symbols_up)
        ychan = self.forward_channel(tx)
        y = self.forward_rx(ychan)
        return y

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

    def post_update(self):
        pass

    def create_lr_schedule(self, optimizer: torch.optim.Optimizer, lr_schedule_str: str,
                           learning_rate: float, n_symbols: int = 0,
                           n_data_chunks: bool | int =False):
        if lr_schedule_str == 'oneclr' and not n_data_chunks:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                               max_lr=10 * learning_rate,
                                                               steps_per_epoch=1,
                                                               epochs=n_symbols // self.batch_size)
        elif lr_schedule_str == 'oneclr' and n_data_chunks:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                               max_lr=10 * learning_rate,
                                                               steps_per_epoch=1,
                                                               epochs=n_data_chunks)
        elif lr_schedule_str == 'expdecay' and not n_data_chunks:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                  gamma=0.99)
        elif lr_schedule_str == 'expdecay' and n_data_chunks:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                  gamma=0.5)
        elif lr_schedule_str == 'multistep':
            # TODO: Should we create a multistep for chunked_data?
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[int(0.5*n_symbols // self.batch_size),
                                                                            int(0.9*n_symbols // self.batch_size)],
                                                                gamma=0.1)
        else:
            raise ValueError(f"Unknown supplied learning rate scheduler: {lr_schedule_str}")

        return lr_scheduler


    def optimize(self, symbols: npt.ArrayLike, return_loss=False):
        loss_array = self.optimize_method_funcp(symbols)
        if return_loss:
            return loss_array

    def _optimize_backprop_funcp(self, symbols: npt.ArrayLike):
        symbols_up = np.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols

        loss_per_batch = np.empty((len(symbols) // self.batch_size, ), dtype=np.float64)

        # Create learning rate scheduler
        if self.lr_schedule:
            lr_scheduler = self.create_lr_schedule(self.optimizer, lr_schedule_str=self.lr_schedule,
                                                   n_symbols=len(symbols), learning_rate=self.learning_rate)

        for b in range(len(symbols) // self.batch_size):
            # Zero gradients
            self.optimizer.zero_grad()

            # Slice out batch and create tensors
            this_a_up = symbols_up[b * self.batch_size * self.sps:(b * self.batch_size * self.sps + self.batch_size * self.sps)]
            target = torch.from_numpy(symbols[b * self.batch_size:(b * self.batch_size + self.batch_size)])
            tx_syms_up = torch.from_numpy(this_a_up)

            # Run upsampled symbols through system forward model - return symbols at Rx
            tx = self.forward_tx(tx_syms_up)
            ychan = self.forward_channel(tx)
            rx_out = self.forward_rx(ychan)

            # Calculate loss
            loss = self.calculate_loss(target, rx_out)

            # Update model using backpropagation
            loss.backward()

            if not self.optimizer:
                raise Exception("Optimizer was not initialized. Please call the 'initialize_optimizer' method before proceeding to optimize.")

            # Gradient norm clipping
            if self.use_gradient_norm_clipping:
                for pgroup in self.optimizer.param_groups:
                    # FIXME: Abusing param groups a bit here.
                    # So far each param group corresponds to exatcly one parameter.
                    torch.nn.utils.clip_grad_norm_(pgroup['params'], 1.0)  # clip all gradients to unit norm

            # Take gradient step.
            self.optimizer.step()
            self.post_update()

            this_lr = self.optimizer.param_groups[-1]['lr']
            if self.lr_schedule:
                lr_scheduler.step()
                this_lr = lr_scheduler.get_last_lr()[0]

            if b % self.batch_print_interval == 0:
                print(f"Batch {b} (# symbols {b * self.batch_size:.2e}) - Loss: {loss.item():.3f} - LR: {this_lr:.2e}")

            loss_per_batch[b] = loss.item()

            if torch.isnan(loss):
                print("Detected loss to be nan. Terminate training...")
                break

        return loss_per_batch

    def _optimize_surrogate_funcp(self, symbols: npt.ArrayLike):
        n_syms_pr_chunk = int(len(symbols) * self.surrogate_chunk_size_pct)
        symbol_losses = []

        symbols_tensor = torch.from_numpy(symbols)

        # Create learning rate scheduler(s)
        if self.surrogate_lr_schedule:
            surrogate_lr_scheduler = self.create_lr_schedule(self.surrogate_optimizer, lr_schedule_str=self.surrogate_lr_schedule,
                                                             n_symbols=len(symbols), learning_rate=self.surrogate_learning_rate,
                                                             n_data_chunks=len(symbols)//n_syms_pr_chunk)

        if self.lr_schedule and (self.learn_rx or not self.tx_learning_rate):
            lr_scheduler = self.create_lr_schedule(self.optimizer, lr_schedule_str=self.lr_schedule,
                                                   n_symbols=len(symbols), learning_rate=self.learning_rate,
                                                   n_data_chunks=len(symbols)//n_syms_pr_chunk)
        if self.tx_lr_schedule and self.tx_learning_rate:
            tx_lr_schedule = self.create_lr_schedule(self.tx_optimizer, lr_schedule_str=self.tx_lr_schedule,
                                                     n_symbols=len(symbols), learning_rate=self.tx_learning_rate,
                                                     n_data_chunks=len(symbols)//n_syms_pr_chunk)

        # Loop over the chunks of the data
        for chunk in range(len(symbols) // n_syms_pr_chunk):
            # Optimization step on the the surrogate channel
            # NB! Always start with the surrogate channel - needs a head start to converge before updating anything else.
            for param in self.surrogate_channel.parameters():
                param.requires_grad = True
            surrogate_loss = self._optimize_surrogate_channel_only(symbols_tensor[chunk*n_syms_pr_chunk:(chunk*n_syms_pr_chunk + n_syms_pr_chunk)])

            # Fix surrogate and optimize filters.
            for param in self.surrogate_channel.parameters():
                param.requires_grad = False
            rx_symbol_loss, tx_symbol_loss = self._optimize_surrogate_full(symbols_tensor[chunk*n_syms_pr_chunk:(chunk*n_syms_pr_chunk + n_syms_pr_chunk)])
            symbol_losses.append(rx_symbol_loss)

            # Update stepsizes according to schedule
            if self.surrogate_lr_schedule:
                surrogate_lr_scheduler.step()

            if self.lr_schedule and (self.learn_rx or not self.tx_learning_rate):
                lr_scheduler.step()

            if self.tx_lr_schedule and self.tx_learning_rate:
                tx_lr_schedule.step()
            
            # Print the status of the optimization (loss and lrs)
            this_surrogate_lr = self.surrogate_optimizer.param_groups[-1]['lr']
            this_rx_lr = self.optimizer.param_groups[-1]['lr']
            this_tx_lr = self.tx_optimizer.param_groups[-1]['lr'] if self.tx_learning_rate else np.nan
            print_strs = [f"Chunk {chunk} (# symbols {chunk * n_syms_pr_chunk:.2e})"]
            print_strs.append(f"Surrogate loss: {surrogate_loss[-1]:.3f} (LR: {this_surrogate_lr:.2e})")
            print_strs.append(f"Symbol loss (Rx): {rx_symbol_loss[-1]:.3f} (LR: {this_rx_lr:.2e})")
            print_strs.append(f"Symbol loss (Tx): {tx_symbol_loss[-1]:.3f} (LR: {this_tx_lr:.2e})")
            print(" - ".join(print_strs))

        return np.concatenate(symbol_losses)

    def _optimize_surrogate_channel_only(self, symbols: torch.TensorType):
        symbols_up = torch.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols

        loss_per_batch = np.empty((len(symbols) // self.batch_size, ), dtype=np.float64)

        for b in range(len(symbols) // self.batch_size):
            # Slice out batch and create tensors
            tx_syms_up = symbols_up[b * self.batch_size * self.sps:(b * self.batch_size * self.sps + self.batch_size * self.sps)]

            # Gradient descent on surrogate model - use real channel as ground truth
            self.surrogate_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                tx = self.forward_tx(tx_syms_up)
                ychan_true = self.forward_channel(tx)
            ychan = self.surrogate_channel.forward(tx)
            surrogate_loss = self.calculate_surrogate_loss(ychan, ychan_true)
            surrogate_loss.backward()
            self.surrogate_optimizer.step()

            loss_per_batch[b] = surrogate_loss.item()

            if torch.isnan(surrogate_loss):
                print("Detected loss to be nan. Terminate training...")
                break

        return loss_per_batch

    def _optimize_surrogate_full(self, symbols: torch.TensorType):
        if not self.optimizer:
            raise Exception("Optimizer was not initialized. Please call the 'initialize_optimizer' method before proceeding to optimize.")

        symbols_up = torch.zeros(self.sps * len(symbols), dtype=symbols.dtype)
        symbols_up[0::self.sps] = symbols

        rx_loss_per_batch = np.empty((len(symbols) // self.batch_size, ), dtype=np.float64)
        tx_loss_per_batch = np.empty((len(symbols) // self.batch_size, ), dtype=np.float64)

        for b in range(len(symbols) // self.batch_size):
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            if self.tx_learning_rate:
                self.tx_optimizer.zero_grad(set_to_none=True)

            # Slice out batch
            target = symbols[b * self.batch_size:(b * self.batch_size + self.batch_size)]
            tx_syms_up = symbols_up[b * self.batch_size * self.sps:(b * self.batch_size * self.sps + self.batch_size * self.sps)]

            # Run upsampled symbols through system forward model - return symbols at Rx
            # Run Tx signal through real channel
            with torch.no_grad():
                tx = self.forward_tx(tx_syms_up)
                ychan_true = self.forward_channel(tx)

            # Then use the surrogate for updating the Tx params
            tx = self.forward_tx(tx_syms_up)
            ychan = self.surrogate_channel.forward(tx)
            rx_out_surrogate = self.forward_rx(ychan)
            tx_loss = self.calculate_loss(target, rx_out_surrogate)

            # Furthermore, add the "Rx-only-loss" that uses the real-channel
            rx_out = self.forward_rx(ychan_true)

            # Calculate loss
            rx_loss = self.calculate_loss(target, rx_out)

            if self.tx_learning_rate:
                # Update using backpropagation
                tx_loss.backward()

                # Gradient norm clipping
                if self.use_gradient_norm_clipping:
                    for pgroup in self.get_tx_parameters():
                        torch.nn.utils.clip_grad_norm_(pgroup['params'], 1.0)  # clip all gradients to unit norm

                # Take gradient step.
                self.tx_optimizer.step()

                # Update using backpropagation
                if self.learn_rx:
                    rx_loss.backward()
                    self.optimizer.step()
            else:
                # Combined optimization
                loss = tx_loss + rx_loss
                loss.backward()

                # Gradient norm clipping
                if self.use_gradient_norm_clipping:
                    for pgroup in self.get_tx_parameters():
                        torch.nn.utils.clip_grad_norm_(pgroup['params'], 1.0)  # clip all gradients to unit norm

                self.optimizer.step()
            
            self.post_update()

            rx_loss_per_batch[b] = rx_loss.item()
            tx_loss_per_batch[b] = tx_loss.item()

            if torch.isnan(rx_loss) or torch.isnan(tx_loss):
                print("Detected loss to be nan. Terminate training...")
                break

        return rx_loss_per_batch, tx_loss_per_batch


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
                 tx_filter_init_type='dirac', rx_filter_init_type='dirac', rrc_rolloff=0.5, lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps,
                         esn0_db=esn0_db,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_rx=learn_rx,
                         learn_tx=learn_tx,
                         print_interval=print_interval,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)

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

    def get_rx_parameters(self):
        return [{"params": self.rx_filter.parameters()}]

    def get_tx_parameters(self):
        return [{"params": self.pulse_shaper.parameters()}]

    def forward_tx(self, x_syms_up: torch.TensorType):
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(x_syms_up)

        # Normalize (if self.normalization_after_tx is set, else norm_constant = 1.0)
        x = x / self.normalization_constant
        return x

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
         # Add white noise based on desired EsN0
        with torch.no_grad():
            noise_std = self.calculate_noise_std(tx_signal)

        y = tx_signal + noise_std * torch.randn(tx_signal.shape)
        return y

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        # Apply rx filter
        rx_filter_out = self.rx_filter.forward(y_channel)

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

    def post_update(self):
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
                 normalize_after_tx=True, filter_init_type='dirac', lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
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
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)


class RxFilteringAWGN(BasicAWGN):
    """
        Special case of the BasicAWGN model learning the Rx filter (Tx filter set to RRC)
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate, rx_filter_length,
                 tx_filter_length, print_interval=int(50000), rrc_rolloff=0.5,
                 normalize_after_tx=True, filter_init_type='dirac', lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         learn_tx=False, learn_rx=True,
                         tx_filter_length=tx_filter_length,
                         rx_filter_length=rx_filter_length,
                         print_interval=print_interval,
                         rrc_rolloff=rrc_rolloff,
                         normalize_after_tx=normalize_after_tx,
                         rx_filter_init_type=filter_init_type,
                         tx_filter_init_type='rrc',
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)


class BasicAWGNwithBWL(LearnableTransmissionSystem):
    """
        Basic additive white Gaussian noise system with pulse-shaping by RRC and bandwidth limitation
        Bandwidth limitation parameter is defined relative to the bandwidth of the RRC pulse.
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation,
                 tx_filter_length: int, rx_filter_length: int, adc_bwl_relative_cutoff,
                 dac_bwl_relative_cutoff, learn_tx: bool, learn_rx: bool, equaliser_config: dict | None = None,
                 tx_filter_init_type='dirac', rx_filter_init_type='dirac',
                 print_interval=int(5e4), lp_filter_type='bessel', lr_schedule='oneclr',
                 rrc_rolloff=0.5, tx_multi_channel: bool = False,
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps,
                         esn0_db=esn0_db,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_rx=learn_rx,
                         learn_tx=learn_tx,
                         print_interval=print_interval,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params,
                         tx_multi_channel=tx_multi_channel)

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

        self.pulse_shaper = AllPassFilter()
        if tx_multi_channel:
            self.pulse_shaper = MultiChannelFIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)
        else:
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
            self.equaliser = LinearFeedForwardEqualiser(samples_per_symbol=self.sps, dtype=torch.float64,
                                                        **equaliser_config)

        # Define bandwidth limitation filters - low pass filter with cutoff relative to bandwidth of baseband
        info_bw = 0.5 * baud_rate

        # Digital-to-analog (DAC) converter
        self.dac = AllPassFilter()
        if dac_bwl_relative_cutoff is not None:
            if lp_filter_type == 'fir':
                if tx_multi_channel:
                    raise NotImplementedError()
                else:
                    self.dac = LowPassFIR(num_taps=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            elif lp_filter_type == 'bessel':
                if tx_multi_channel:
                    self.dac = MultiChannelBesselFilter(bessel_order=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
                else:
                    self.dac = BesselFilter(bessel_order=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                raise ValueError(f"Unknown low-pass filter type: {lp_filter_type}")

        # Analog-to-digial (ADC) converter
        self.adc = AllPassFilter()

        if adc_bwl_relative_cutoff is not None:
            if lp_filter_type == 'fir':
                self.adc = LowPassFIR(num_taps=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            elif lp_filter_type == 'bessel':
                self.adc = BesselFilter(bessel_order=5, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                raise ValueError(f"Unknown low-pass filter type: {lp_filter_type}")

        self.normalization_constant = np.sqrt(np.average(np.square(self.constellation)) / self.sps)

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int((self.pulse_shaper.filter_length + self.rx_filter.filter_length) / self.sps)

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

        # Total symbol delay introduced by the two LPFs
        self.channel_delay_in_syms = int(np.ceil((self.adc.get_sample_delay() + self.dac.get_sample_delay())/self.sps))
        print(f"Channel delay is {self.channel_delay_in_syms} [symbols]")

    def get_rx_parameters(self):
        params_to_return =  [{"params": self.rx_filter.parameters()}]
        if self.use_eq:
            params_to_return += self.equaliser.get_parameters()
        return params_to_return

    def get_tx_parameters(self):
        return [{"params": self.pulse_shaper.parameters()}]

    def forward_tx(self, x_syms_up: torch.TensorType) -> torch.TensorType:
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(x_syms_up)

        return x

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
        # Normalize
        x = tx_signal / self.normalization_constant

        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward(x)

        # Add white noise
        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp)
        y = x_lp + noise_std * torch.randn(x_lp.shape)

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y)

        return  y_lp

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        # Apply rx filter
        rx_filter_out = self.rx_filter.forward(y_channel)

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
        return torch.mean(torch.square(torch.subtract(tx_syms[self.discard_per_batch:-self.discard_per_batch],
                                                      torch.roll(rx_syms, -self.channel_delay_in_syms)[self.discard_per_batch:-self.discard_per_batch])))

    def post_update(self):
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
                 lp_filter_type='bessel', lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)


class RxFilteringAWGNwithBWL(BasicAWGNwithBWL):
    """
       Bandwidth limited AWGN channel with learnable Rx filter.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 filter_init_type='dirac', print_interval=int(5e4), rrc_rolloff=0.5,
                 lp_filter_type='bessel', lr_schedule='oneclr') -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=None)  # not optimizing the Tx


class JointTxRxAWGNwithBWL(BasicAWGNwithBWL):
    """
       Bandwidth limited AWGN channel with learnable Tx and Rx filter.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5,
                 lp_filter_type='bessel', lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)


class LinearFFEAWGNwithBWL(BasicAWGNwithBWL):
    """
       Bandwidth limited AWGN channel with fixed Tx and Rx filters.
       Adaptive FFE equaliser to combat ISI.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 ffe_n_taps, rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5, lp_filter_type='bessel',
                 lr_schedule='oneclr') -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=None)  # not optimizing the transmitter

    def get_equaliser_filter(self):
        return self.equaliser.filter.get_filter()


class BasicAWGNwithBWLandWDM(BasicAWGNwithBWL):
    """
        Bandwidth limited AWGN channel with wavelength division multiplexing (WDM)

        TODO: How to integrate surrogate model in WDM? How much should it model?
        WIP...
         - Only model the channel of interest? (How to structure that in code?)
         - MultiChannel input, single output?
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size, constellation,
                 tx_filter_length: int, rx_filter_length: int, adc_bwl_relative_cutoff,
                 dac_bwl_relative_cutoff, learn_tx: bool, learn_rx: bool,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 equaliser_config: dict | None = None, tx_filter_init_type='dirac',
                 rx_filter_init_type='dirac', print_interval=int(50000), torch_seed=0,
                 lp_filter_type='bessel', lr_schedule='oneclr', rrc_rolloff=0.5,
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, esn0_db=esn0_db, baud_rate=baud_rate,
                         learning_rate=learning_rate, batch_size=batch_size,
                         constellation=constellation,
                         tx_filter_length=tx_filter_length,
                         rx_filter_length=rx_filter_length,
                         adc_bwl_relative_cutoff=adc_bwl_relative_cutoff,
                         dac_bwl_relative_cutoff=dac_bwl_relative_cutoff,
                         learn_tx=learn_tx, learn_rx=learn_rx,
                         equaliser_config=equaliser_config,
                         tx_filter_init_type=tx_filter_init_type,
                         rx_filter_init_type=rx_filter_init_type,
                         print_interval=print_interval,
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         rrc_rolloff=rrc_rolloff,
                         tx_multi_channel=True,
                         tx_optimizer_params=tx_optimizer_params)

        self.wdm_n_channels = 3  # always three channels during training
        self.wdm_channel_spacing_hz = wdm_channel_spacing_hz
        self.torch_seed = torch_seed  # used for generating interferer channels

        # Create Gauss filter object for channel selection on Rx side
        self.channel_selection_filter = GaussianFqFilter(filter_cutoff_hz=(0.5 * self.baud_rate) * wdm_channel_selection_rel_cutoff,
                                                         order=5,
                                                         Fs=1/self.Ts)

    def forward_tx(self, x_syms_up: torch.TensorType) -> torch.TensorType:
        # Prepare WDM channel
        rng_gen = torch.random.manual_seed(self.torch_seed)
        symbols_up_chan = torch.stack((permute_symbols(x_syms_up, self.sps, rng_gen),
                                       x_syms_up,
                                       permute_symbols(x_syms_up, self.sps, rng_gen)), dim=1)

        # Apply same pulse shaper to all channels (MultiChannelFIRfilter)
        x = self.pulse_shaper.forward(symbols_up_chan)
        x = x / self.normalization_constant

        return x

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
        # Apply bandwidth limitation in the DAC (multi-channel as well)
        x_lp = self.dac.forward(tx_signal)

        # Create WDM signal
        channel_fq_grid = torch.Tensor([-1.0, 0.0, 1.0]) * self.wdm_channel_spacing_hz
        tx_wdm = torch.sum(x_lp * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[None, :] * self.Ts) * torch.arange(0, x_lp.shape[0], dtype=torch.float64)[:, None]), dim=1)

        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp[:, 1])  # just calculate the noise_std based on the channel of interest
        y = tx_wdm + noise_std * torch.randn(tx_wdm.shape)

        # Low-pass filter to select middle channel
        y_chan = torch.real(self.channel_selection_filter.forward(y))

        # Apply bandwidth limitation in the ADC
        y_lp = self.adc.forward(y_chan)

        return y_lp

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        # Apply rx filter - applies stride inside filter (outputs sps = 1, if no equaliser)
        rx_filter_out = self.rx_filter.forward(y_channel)

        # Apply equaliser
        rx_eq_out = self.equaliser.forward(rx_filter_out)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    def _eval(self, symbols_up: torch.TensorType, batch_size: int, decimate: bool = True):
        # Prepare WDM channel
        rng_gen = torch.random.manual_seed(self.torch_seed)
        channel_fq_grid = torch.Tensor([-1.0, 0.0, 1.0]) * self.wdm_channel_spacing_hz

        symbols_up_chan = torch.stack((permute_symbols(symbols_up, self.sps, rng_gen),
                                       symbols_up,
                                       permute_symbols(symbols_up, self.sps, rng_gen)), dim=1)

        # Apply same pulse shaper to all channels (MultiChannelFIRfilter)
        x = self.pulse_shaper.forward_numpy(symbols_up_chan)
        x = x / self.normalization_constant

        # Apply bandwidth limitation in the DAC (multi-channel as well)
        x_lp = self.dac.forward_numpy(x)

        tx_wdm = torch.sum(x_lp * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[None, :] * self.Ts) * torch.arange(0, x_lp.shape[0], dtype=torch.float64)[:, None]), dim=1)

        with torch.no_grad():
            noise_std = self.calculate_noise_std(x_lp[:, 1])  # just calculate the noise_std based on the channel of interest
        y = tx_wdm + noise_std * torch.randn(tx_wdm.shape)

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
                 lp_filter_type='bessel', lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)


class RxFilteringAWGNwithBWLandWDM(BasicAWGNwithBWLandWDM):
    """
       Bandwidth limited AWGN channel with learnable Rx filter and WDM.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 filter_init_type='dirac', print_interval=int(5e4), rrc_rolloff=0.5,
                 lp_filter_type='bessel', lr_schedule='oneclr') -> None:
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=None)


class JointTxRxAWGNwithBWLandWDM(BasicAWGNwithBWLandWDM):
    """
       Bandwidth limited AWGN channel (WDM) with learnable Tx and Rx filter.
    """
    def __init__(self, sps, esn0_db, baud_rate, constellation, batch_size, learning_rate,
                 rx_filter_length, tx_filter_length, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 print_interval=int(5e4), rrc_rolloff=0.5,
                 lp_filter_type='bessel', lr_schedule='oneclr',
                 tx_optimizer_params: dict | None = None) -> None:
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)


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
                 lp_filter_type='bessel', lr_schedule='oneclr') -> None:
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
                         lp_filter_type=lp_filter_type,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=None)

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
                 print_interval=int(5e4), lp_filter_type='bessel', lr_schedule='oneclr',
                 rrc_rolloff=0.5, tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps,
                         esn0_db=esn0_db,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_rx=learn_rx,
                         learn_tx=learn_tx,
                         print_interval=print_interval,
                         lr_schedule=lr_schedule,
                         tx_optimizer_params=tx_optimizer_params)

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

        # TODO: Implement WDM for this NonLinearISI channel
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
            if lp_filter_type == 'fir':
                self.dac = LowPassFIR(num_taps=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            elif lp_filter_type == 'bessel':
                self.dac = BesselFilter(bessel_order=5, cutoff_hz=info_bw * dac_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                raise ValueError(f"Unknown low-pass filter type: {lp_filter_type}")

        # Analog-to-digial (ADC) converter
        self.adc = AllPassFilter()
        if adc_bwl_relative_cutoff is not None:
            if lp_filter_type == 'fir':
                self.adc = LowPassFIR(num_taps=5, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)
            elif lp_filter_type == 'bessel':
                self.adc = BesselFilter(bessel_order=5, cutoff_hz=info_bw * adc_bwl_relative_cutoff, fs=1/self.Ts)
            else:
                raise ValueError(f"Unknown low-pass filter type: {lp_filter_type}")

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
        self.channel_delay_in_syms = int(np.ceil((isi_delay + self.adc.get_sample_delay() + self.dac.get_sample_delay()) / sps))
        print(f"Channel delay is {self.channel_delay_in_syms} [symbols] (ISI contributed with {int(np.ceil(isi_delay / sps))})")

        # Calculate constellation scale
        self.constellation_scale = np.sqrt(np.average(np.square(constellation)))

    def get_rx_parameters(self):
        params_to_return =  [{"params": self.rx_filter.parameters()}]
        return params_to_return

    def get_tx_parameters(self):
        return [{"params": self.pulse_shaper.parameters()}]

    def forward_tx(self, x_syms_up: torch.TensorType) -> torch.TensorType:
        # Input is assumed to be upsampled sybmols
        # Apply pulse shaper
        x = self.pulse_shaper.forward(x_syms_up)

        # Normalize
        x = x / self.normalization_constant

        return x

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
        # Apply bandwidth limitation in the DAC
        x_lp = self.dac.forward(tx_signal)

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

        return y_lp

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        rx_filter_out = self.rx_filter.forward(y_channel)

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
        rx_filter_out = self.rx_filter.forward_numpy(y_lp)

        # Power normalize and rescale to constellation
        rx_filter_out = rx_filter_out / torch.sqrt(torch.mean(torch.square(rx_filter_out))) * self.constellation_scale

        return rx_filter_out

    def calculate_loss(self, tx_syms: torch.TensorType, rx_syms: torch.TensorType):
        # NB! Rx sequence is coarsely aligned to the tx-symbol sequence based on a apriori known channel delay.
        return torch.mean(torch.square(tx_syms[self.discard_per_batch:-self.discard_per_batch] - torch.roll(rx_syms, -self.channel_delay_in_syms)[self.discard_per_batch:-self.discard_per_batch]))

    def post_update(self):
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
                 filter_init_type='rrc', print_interval=int(50000), lp_filter_type='bessel',
                 lr_schedule='oneclr', rrc_rolloff=0.5,
                 tx_optimizer_params: dict | None = None) -> None:
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
                         print_interval=print_interval, lp_filter_type=lp_filter_type, lr_schedule=lr_schedule,
                         rrc_rolloff=rrc_rolloff,
                         tx_optimizer_params=tx_optimizer_params)


class RxFilteringNonLinearISIChannel(NonLinearISIChannel):
    """
        Learning the Rx filter in the non-linear isi channel
    """
    def __init__(self, sps, esn0_db, baud_rate, learning_rate, batch_size,
                 constellation, adc_bwl_relative_cutoff, dac_bwl_relative_cutoff,
                 rx_filter_length, tx_filter_length, non_linear_coefficients=(0.95, 0.04, 0.01),
                 isi_filter1=np.array([0.2, -0.1, 0.9, 0.3]), isi_filter2=np.array([0.2, 0.9, 0.3]),
                 filter_init_type='rrc', print_interval=int(50000),
                 lp_filter_type='bessel', lr_schedule='oneclr', rrc_rolloff=0.5) -> None:
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
                         print_interval=print_interval, lp_filter_type=lp_filter_type, lr_schedule=lr_schedule,
                         rrc_rolloff=rrc_rolloff,
                         tx_optimizer_params=None)


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
                 rx_filter_init_type='rrc', print_interval=int(50000), lp_filter_type='bessel',
                 lr_schedule='oneclr', rrc_rolloff=0.5,
                 tx_optimizer_params: dict | None = None) -> None:
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
                         print_interval=print_interval, lp_filter_type=lp_filter_type, lr_schedule=lr_schedule,
                         rrc_rolloff=rrc_rolloff,
                         tx_optimizer_params=tx_optimizer_params)


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
                                                        channel (SMF or SSFM)
                                                          |
        symbols hat <-symbol norm <- filtering <- adc <- photodiode

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 learn_rx, learn_tx, rx_filter_length, tx_filter_length,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz: float | None, modulator_type='eam',
                 equaliser_config: dict | None = None, fiber_type='smf',
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 rrc_rolloff=0.5, adc_bitres=None, dac_minmax_norm: float | str = 'auto',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 multi_channel: bool=False, adc_lp_filter_type='bessel',
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps,
                         esn0_db=np.nan,
                         baud_rate=baud_rate,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         constellation=constellation,
                         learn_rx=learn_rx,
                         learn_tx=learn_tx,
                         lr_schedule=lr_schedule,
                         eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval,
                         tx_optimizer_params=tx_optimizer_params)

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

        self.pulse_shaper = AllPassFilter()
        if multi_channel:
            self.pulse_shaper = MultiChannelFIRfilter(filter_weights=tx_filter_init, trainable=learn_tx)
        else:
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
            equaliser_type = equaliser_config.pop('type', 'ffe')
            if equaliser_type.lower() == 'ffe':
                self.equaliser = LinearFeedForwardEqualiser(samples_per_symbol=self.sps, dtype=torch.float64,
                                                            **equaliser_config)
            elif equaliser_type.lower() == 'volterra':
                self.equaliser = VolterraEqualizer(samples_per_symbol=self.sps, dtype=torch.float64,
                                                   **equaliser_config)
            else:
                raise ValueError(f"Uknown equaliser type: '{equaliser_type}'")

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
        self.optimizable_dac = dac_config.get('learnable_bias') or dac_config.get('learnable_normalization')
        self.dac = DigitalToAnalogConverter(peak_to_peak_constellation=dac_normalizer,
                                            multi_channel=multi_channel,
                                            fs=1/self.Ts,
                                            **dac_config)

        # Analog-to-digial (ADC) converter
        self.adc = AnalogToDigitalConverter(bwl_cutoff=adc_bwl_cutoff_hz, fs=1/self.Ts,
                                            bit_resolution=adc_bitres,
                                            filter_type=adc_lp_filter_type)

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
            raise ValueError(f"Unknown modulator type '{modulator_type}'. Valid options are: 'ideal', 'eam', 'mzm' or 'nonlin_eam'")

        # Define channel (fiber model)
        self.channel = AllPassFilter()
        self.fiber_type = fiber_type.lower()
        if self.fiber_type == "smf":
            self.channel = SingleModeFiber(Fs=1/self.Ts, **fiber_config)
        elif self.fiber_type == "ssfm":
            self.channel = SplitStepFourierFiber(Fs=1/self.Ts, **fiber_config)
        else:
            raise ValueError(f"Unknown fiber type '{fiber_type}'. Valid options are: 'smf' or 'ssfm'")

        # Define photodiode
        self.photodiode = Photodiode(bandwidth=adc_bwl_cutoff_hz if adc_bwl_cutoff_hz is not None else baud_rate * 0.5,
                                     Fs=1/self.Ts, sps=self.sps, **photodiode_config)
        self.Es = None  # initialize energy-per-symbol to None as it will be calculated on the fly during eval

        # Define number of symbols to discard pr. batch due to boundary effects of convolution
        self.discard_per_batch = int(((self.pulse_shaper.filter_length + self.rx_filter.filter_length) // 2) / self.sps)

        # Calculate estimate of channel delay (in symbols)
        self.channel_delay_in_syms = int(np.ceil((self.adc.get_sample_delay() + self.dac.get_sample_delay()) / sps))
        print(f"Channel delay is {self.channel_delay_in_syms} [symbols]")

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

    def get_rx_parameters(self):
        params_to_return =  [{"params": self.rx_filter.parameters()}]
        if self.use_eq:
            params_to_return += self.equaliser.get_parameters()
        return params_to_return

    def get_tx_parameters(self):
        params_to_return = [{"params": self.pulse_shaper.parameters()}]
        if self.optimizable_dac:
            params_to_return.append({"params": self.dac.parameters()})
        return params_to_return

    def forward_tx(self, x_syms_up: torch.TensorType) -> torch.TensorType:
        # Apply pulse shaper
        x = self.pulse_shaper.forward(x_syms_up)

        return x

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
        # Apply bandwidth limitation in the DAC
        v = self.dac.forward(tx_signal)

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

        return y_norm

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        # Apply rx filter - applies stride inside filter (outputs sps = 1)
        # (if equaliser is not specified)
        rx_filter_out = self.rx_filter.forward(y_channel)

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
        laser_power_dbm = eval_config.get('laser_power_dbm', None)
        if laser_power_dbm:
            self.modulator.set_laser_power_dbm(laser_power_dbm)
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
        print(f"Photodiode (after ADC and norm): Noise variance: {self.photodiode.get_total_noise_variance() / y_lp.std()}")

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
        return torch.mean(torch.square(tx_syms[self.discard_per_batch:-self.discard_per_batch] - torch.roll(rx_syms, -self.channel_delay_in_syms)[self.discard_per_batch:-self.discard_per_batch]))

    def post_update(self):
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
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz, modulator_type='eam', fiber_type='smf',
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc', rrc_rolloff=0.5,
                 dac_minmax_norm: float | str = 'auto', adc_bitres=None, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         learn_rx=False, learn_tx=True, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length, fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         modulator_type=modulator_type, dac_config=dac_config,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz,
                         adc_bitres=adc_bitres, dac_minmax_norm=dac_minmax_norm,
                         rrc_rolloff=rrc_rolloff, lr_schedule=lr_schedule, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval,
                         adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=tx_optimizer_params)


class RxFilteringIM(IntensityModulationChannel):
    """
        RxFiltering (learning Rx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 rrc_rolloff=0.5, adc_bitres=None, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         dac_minmax_norm=dac_minmax_norm, dac_config=dac_config,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         modulator_type=modulator_type, learn_rx=True, learn_tx=False,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, lr_schedule=lr_schedule, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval,
                         adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=None)

class JointTxRxIM(IntensityModulationChannel):
    """
        JointTxRx (learning both Tx andRx filter) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz: float, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 rrc_rolloff=0.5, adc_bitres=None, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         modulator_type=modulator_type, dac_config=dac_config,
                         dac_minmax_norm=dac_minmax_norm, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz,
                         learn_rx=True, learn_tx=True, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, lr_schedule=lr_schedule, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval,
                         adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=tx_optimizer_params)


class LinearFFEIM(IntensityModulationChannel):
    """
        RRC + Matched filter + Linear FFE in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 rrc_rolloff=0.5, adc_bitres=None, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, dac_minmax_norm=dac_minmax_norm,
                         modulator_type=modulator_type, equaliser_config={'n_taps': ffe_n_taps},
                         learn_rx=False, learn_tx=False, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, lr_schedule=lr_schedule, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval,
                         adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=None)

    def get_equaliser_filter(self):
        return self.equaliser.filter.get_filter()

class VolterraIM(IntensityModulationChannel):
    """
        RRC + Matched filter + Volterra (FFE) in the (Liang and Kahn, 2023) IM/DD system
    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps1, ffe_n_taps2,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 rrc_rolloff=0.5, adc_bitres=None, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000)) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, dac_minmax_norm=dac_minmax_norm,
                         modulator_type=modulator_type, equaliser_config={'type': 'volterra', 'n_lags1': ffe_n_taps1,
                                                                          'n_lags2': ffe_n_taps2},
                         learn_rx=False, learn_tx=False, rx_filter_length=rx_filter_length,
                         tx_filter_length=tx_filter_length,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_bitres=adc_bitres,
                         rrc_rolloff=rrc_rolloff, lr_schedule=lr_schedule, eval_batch_size_in_syms=eval_batch_size_in_syms,
                         print_interval=print_interval,
                         adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=None)

    def get_equaliser_filter(self):
        raise NotImplementedError



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

        [tx] x n_channels -> WDM shift and add ->  singe mode fiber / split-step fourier
                                                          |
                                                    channel selection
                                                      (filtering)
                                                          |
          <-  symbol decision <- filtering <- adc <- photodiode

          TODO: How to implement surrogate channel optimization with WDM?

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 learn_rx, learn_tx, rx_filter_length, tx_filter_length,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, wdm_channel_spacing_hz, adc_bwl_cutoff_hz,
                 wdm_channel_selection_rel_cutoff, wdm_n_channels, dac_minmax_norm: float | str = 'auto',
                 modulator_type='eam', fiber_type='smf', equaliser_config: dict | None = None,
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 adc_lp_filter_type='bessel',
                 rrc_rolloff=0.5, lr_schedule='oneclr', eval_batch_size_in_syms=1000,
                 print_interval=int(50000), torch_seed=0,
                 tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=learn_rx, learn_tx=learn_tx, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, dac_minmax_norm=dac_minmax_norm,
                         modulator_type=modulator_type, equaliser_config=equaliser_config,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, rrc_rolloff=rrc_rolloff, adc_bitres=None,
                         eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         multi_channel=True, adc_lp_filter_type=adc_lp_filter_type, tx_optimizer_params=tx_optimizer_params)

        self.wdm_n_channels = wdm_n_channels
        assert (wdm_n_channels - 1) % 2 == 0
        self.wdm_channel_spacing_hz = wdm_channel_spacing_hz
        self.torch_seed = torch_seed  # used for generating interferer channels

        # Create Gauss filter object for channel selection on Rx side
        self.channel_selection_filter = GaussianFqFilter(filter_cutoff_hz=(0.5 * self.baud_rate) * wdm_channel_selection_rel_cutoff,
                                                         order=5,
                                                         Fs=1/self.Ts)
        
    def _interferer_channels(self, xsyms: torch.TensorType, rng_gen, n_channels):
        # Generate a list of channels with our channel of interest in the middle
        x_channels = []
        x_channels += [permute_symbols(xsyms, self.sps, rng_gen) for __ in range((n_channels - 1) // 2)]
        x_channels += [xsyms]
        x_channels += [permute_symbols(xsyms, self.sps, rng_gen) for __ in range((n_channels - 1) // 2)]
        return x_channels

    def forward_tx(self, x_syms_up: torch.TensorType) -> torch.TensorType:
        # Prepare WDM channel
        rng_gen = torch.random.manual_seed(self.torch_seed)

        symbols_up_chan = torch.stack(self._interferer_channels(x_syms_up, rng_gen, self.wdm_n_channels), dim=1)

        # Apply pulse shaper (to all channels independently)
        x = self.pulse_shaper.forward(symbols_up_chan)

        return x

    def forward_channel(self, tx_signal: torch.TensorType) -> torch.TensorType:
        # Apply bandwidth limitation in the DAC
        v = self.dac.forward(tx_signal)

        # Apply EAM
        x_eam = self.modulator.forward(v)

        # Construct WDM signal
        channel_fq_grid = torch.Tensor(np.arange(-self.wdm_n_channels // 2 + 1, self.wdm_n_channels // 2 + 1, 1.0)) * self.wdm_channel_spacing_hz
        tx_wdm = torch.sum(x_eam * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[None, :] * self.Ts) * torch.arange(0, tx_signal.shape[0], dtype=torch.float64)[:, None]), dim=1)

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

        return y_norm

    def forward_rx(self, y_channel: torch.TensorType) -> torch.TensorType:
        # Apply rx filter - applies stride inside filter (outputs sps = 1, if no equalsier)
        rx_filter_out = self.rx_filter.forward(y_channel)

        # Apply equaliser
        rx_eq_out = self.equaliser.forward(rx_filter_out)

        # Power normalize and rescale to constellation
        rx_eq_out = rx_eq_out / torch.sqrt(torch.mean(torch.square(rx_eq_out))) * self.constellation_scale

        return rx_eq_out

    def eval_tx(self, symbols_up: torch.TensorType, channel_spacing_hz: float,
                batch_size: int, laser_power_dbm: float | None = None,
                wdm_n_channels: int | None = None,
                dac_bitres: int | None = None, torch_seed: int = 0):
        # Set DAC bit resolution
        self.dac.set_bitres(dac_bitres)

        # Prepare WDM channel
        nchans = wdm_n_channels if wdm_n_channels else self.wdm_n_channels

        # Generate interferer symbols - randomly permute the input sequence
        rng_gen = torch.random.manual_seed(torch_seed)
        channel_fq_grid = torch.Tensor(np.arange(-nchans // 2 + 1, nchans // 2 + 1, 1.0)) * channel_spacing_hz

        symbols_up_chan = torch.stack(self._interferer_channels(symbols_up, rng_gen, nchans), dim=1)

        print(f"Channel spacing: {channel_spacing_hz / 1e9} GHz")
        print(f"Channel grid: {channel_fq_grid / 1e9} GHz")

        # Apply pulse shaper (to all channels independently)
        x = self.pulse_shaper.forward(symbols_up_chan)

        # Apply bandwidth limitation in the DAC
        v = self.dac.forward(x)

        # Apply EAM
        if laser_power_dbm:
            self.modulator.set_laser_power_dbm(laser_power_dbm)

        x_eam = self.modulator.forward(v)

        # Construct WDM signal
        tx_wdm = torch.sum(x_eam * torch.exp(1j * 2 * torch.pi * (channel_fq_grid[None, :] * self.Ts) * torch.arange(0, x.shape[0], dtype=torch.float64)[:, None]), dim=1)

        print(f"EAM (channel 1): Power at output {10.0 * np.log10(np.average(np.square(np.absolute(x_eam[:, 1].detach().numpy()))) / 1e-3)} [dBm]")

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
        torch_seed = eval_config.get('seed', self.torch_seed)
        laser_power_dbm = eval_config.get('laser_power_dbm', None)
        wdm_n_channels = eval_config.get('wdm_n_channels', None)
        fiber_gamma = eval_config.get('ssfm_gamma', None)

        # Apply Tx (including generating interferer symbols and WDM signal)
        tx_wdm = self.eval_tx(symbols_up, channel_spacing_hz, batch_size, laser_power_dbm=laser_power_dbm,
                              dac_bitres=eval_config.get('dac_bitres', None),
                              wdm_n_channels=wdm_n_channels, torch_seed=torch_seed)

        # Apply fiber model
        if fiber_gamma:
            if self.fiber_type == "ssfm":
                self.channel.set_gamma(fiber_gamma)
            else:
                raise ValueError(f"Cannot change the non-linearity coefficient gamma for a standard fiber.")

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
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 adc_bwl_cutoff_hz, wdm_n_channels=3, adc_lp_filter_type='bessel', modulator_type='eam',
                 fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_minmax_norm: float | str = 'auto', rrc_rolloff=0.5,
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0, tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=False, learn_tx=True, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, dac_minmax_norm=dac_minmax_norm,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         wdm_n_channels=wdm_n_channels,
                         modulator_type=modulator_type, equaliser_config=None,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed, adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=tx_optimizer_params)


class RxFilteringIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Rx-filter is learned

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, wdm_channel_spacing_hz, adc_bwl_cutoff_hz,
                 wdm_channel_selection_rel_cutoff, wdm_n_channels=3, adc_lp_filter_type='bessel',
                 modulator_type='eam', fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_minmax_norm: float | str = 'auto', rrc_rolloff=0.5,
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=True, learn_tx=False, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, dac_minmax_norm=dac_minmax_norm,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         wdm_n_channels=wdm_n_channels,
                         modulator_type=modulator_type, equaliser_config=None,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed, adc_lp_filter_type=adc_lp_filter_type,
                         tx_optimizer_params=None)


class JointTxRxIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Pulse-shaper and rx-filter is learned

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 adc_bwl_cutoff_hz, wdm_n_channels=3, modulator_type='eam', fiber_type='smf',
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_minmax_norm: float | str = 'auto', rrc_rolloff=0.5, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0, tx_optimizer_params: dict | None = None) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=True, learn_tx=True, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, dac_minmax_norm=dac_minmax_norm,
                         wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         wdm_n_channels=wdm_n_channels,
                         modulator_type=modulator_type, equaliser_config=None,
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_lp_filter_type=adc_lp_filter_type,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed, tx_optimizer_params=tx_optimizer_params)


class LinearFFEIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Pulse-shaper and rx-filter are fixed to RRC and linear equaliser is run afterwards

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps: int,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz,  wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 wdm_n_channels=3,  modulator_type='eam', fiber_type='smf', rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_minmax_norm: float | str = 'auto', rrc_rolloff=0.5, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=False, learn_tx=False, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         wdm_n_channels=wdm_n_channels,
                         modulator_type=modulator_type, equaliser_config={'n_taps': ffe_n_taps},
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_minmax_norm=dac_minmax_norm, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_lp_filter_type=adc_lp_filter_type,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed, tx_optimizer_params=None)


class VolterraIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Pulse-shaper and rx-filter are fixed to RRC and Volterra equaliser is run afterwards

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps1: int, ffe_n_taps2: int,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz,  wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 wdm_n_channels=3, modulator_type='eam', fiber_type='smf',
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_minmax_norm: float | str = 'auto', rrc_rolloff=0.5, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=False, learn_tx=False, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         wdm_n_channels=wdm_n_channels,
                         modulator_type=modulator_type, equaliser_config={'type': 'volterra', 'n_lags1': ffe_n_taps1,
                                                                          'n_lags2': ffe_n_taps2},
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_minmax_norm=dac_minmax_norm, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_lp_filter_type=adc_lp_filter_type,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed, tx_optimizer_params=None)


class PulseShapingVolterraIMwithWDM(IntensityModulationChannelwithWDM):
    """
        Intensity modulation/direct detection (IM/DD) system with WDM evaluation

        Learned pulse-shaper; rx-filter is fixed to RRC; Volterra equaliser is run afterwards

    """
    def __init__(self, sps, baud_rate, learning_rate, batch_size, constellation,
                 rx_filter_length, tx_filter_length, ffe_n_taps1: int, ffe_n_taps2: int,
                 fiber_config: dict, photodiode_config: dict, modulator_config: dict,
                 dac_config: dict, adc_bwl_cutoff_hz,  wdm_channel_spacing_hz, wdm_channel_selection_rel_cutoff,
                 wdm_n_channels=3, modulator_type='eam', fiber_type='smf',
                 rx_filter_init_type='rrc', tx_filter_init_type='rrc',
                 dac_minmax_norm: float | str = 'auto', rrc_rolloff=0.5, adc_lp_filter_type='bessel',
                 lr_schedule='oneclr', eval_batch_size_in_syms=1000, print_interval=int(50000),
                 torch_seed=0) -> None:
        super().__init__(sps=sps, baud_rate=baud_rate, learning_rate=learning_rate,
                         batch_size=batch_size, constellation=constellation, lr_schedule=lr_schedule,
                         learn_rx=False, learn_tx=True, rx_filter_length=rx_filter_length, tx_filter_length=tx_filter_length,
                         fiber_config=fiber_config, fiber_type=fiber_type,
                         photodiode_config=photodiode_config, modulator_config=modulator_config,
                         dac_config=dac_config, wdm_channel_spacing_hz=wdm_channel_spacing_hz,
                         wdm_channel_selection_rel_cutoff=wdm_channel_selection_rel_cutoff,
                         wdm_n_channels=wdm_n_channels,
                         modulator_type=modulator_type, equaliser_config={'type': 'volterra', 'n_lags1': ffe_n_taps1,
                                                                          'n_lags2': ffe_n_taps2},
                         rx_filter_init_type=rx_filter_init_type, tx_filter_init_type=tx_filter_init_type,
                         dac_minmax_norm=dac_minmax_norm, adc_bwl_cutoff_hz=adc_bwl_cutoff_hz, adc_lp_filter_type=adc_lp_filter_type,
                         rrc_rolloff=rrc_rolloff, eval_batch_size_in_syms=eval_batch_size_in_syms, print_interval=print_interval,
                         torch_seed=torch_seed, tx_optimizer_params=None)
