import torch
from torch.nn import functional as torchF
from torchaudio.functional import convolve, lfilter
from torch.fft import fft, ifft, fftshift, ifftshift, fftfreq
import numpy.typing as npt
import numpy as np
from scipy.signal import bessel, group_delay
from scipy.signal import lfilter as scipy_lfilter

# FIXME: Simplify these operations
# FIXME: Switch to torchaudio.functional.convolve - does true convolution

def filter_initialization(init_filter: npt.ArrayLike, init_type: str):
    if init_type == 'dirac':
        init_filter[(len(init_filter) - 1) // 2] = 1.0
    elif init_type == 'randn':
        init_filter = np.random.randn(*init_filter.shape)
        init_filter /= np.linalg.norm(init_filter)
    else:
        raise Exception(f"Unknown initialization type: '{init_type}'")

    return init_filter


class AllPassFilter(torch.nn.Module):
    """ All-pass filter - used to enable/disable filters in marble.systems.
    """
    def forward(self, x):
        return x

    def forward_batched(self, x, batch_size):
        return x
    
    def forward_numpy(self, x):
        return x
    
    def get_filters(self):
        return None, None

    def get_sample_delay(self):
        return 0

class FIRfilter(torch.nn.Module):
    def __init__(self, filter_weights: npt.ArrayLike, stride=1, trainable=False, dtype=torch.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        torch_filter = torch.from_numpy(np.copy(filter_weights))
        self.filter_length = len(filter_weights)
        self.padding = ((self.filter_length - 1) // 2, (self.filter_length - self.filter_length % 2) // 2)
        self.conv_weights = torch.nn.Parameter(torch_filter, requires_grad=trainable)
        self.trainalbe = trainable
        self.stride = stride

    def forward(self, x):
        # input x assumed to be [timesteps, ]
        xpadded = torchF.pad(x, self.padding, mode='constant', value=0.0)[None, None, :]
        f_out = torchF.conv1d(xpadded, torch.flip(self.conv_weights, (0,))[None, None, :],
                              stride=self.stride).squeeze_()
        return f_out

    def forward_batched(self, x, batch_size=2000):
        # Forward pass that uses batching but without boundary effects bewteen batches
        assert batch_size > self.padding[0]
        assert batch_size % self.stride == 0
        # Allocate output
        y = torch.empty((x.shape[0] // self.stride,), dtype=x.dtype)
        n_batches = int(np.ceil(len(x) / batch_size))
        prepad = torch.zeros((self.padding[0]))
        postpad = x[batch_size:(batch_size+self.padding[1])]
        outputs_pr_batch = batch_size // self.stride

        for b in range(n_batches - 2):
            xpadded = torch.concat((prepad,
                                    x[b*batch_size:(b*batch_size + batch_size)],
                                    postpad))[None, None, :]
            y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = torchF.conv1d(xpadded, torch.flip(self.conv_weights, (0,))[None, None, :],
                                                                                          stride=self.stride).squeeze_()

            # Update prepad and postpad
            prepad = x[(b+1)*batch_size-self.padding[0]:(b+1)*batch_size]
            postpad = x[((b + 2)*batch_size):((b + 2)*batch_size + self.padding[1])]

        # Calcualte second to last batch
        b = n_batches - 2
        xpadded = torch.concat((prepad,
                                x[b*batch_size:(b*batch_size + batch_size)],
                                postpad))[None, None, :]
        y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = torchF.conv1d(xpadded, torch.flip(self.conv_weights, (0,))[None, None, :],
                                                                                      stride=self.stride).squeeze_()

        # For the last batch postpad with zeros
        prepad = x[(b+1)*batch_size-self.padding[0]:(b+1)*batch_size]
        b = n_batches - 1
        xpadded = torch.concat((prepad,
                                x[b*batch_size:(b*batch_size + batch_size)],
                                torch.zeros((self.padding[1]))))[None, None, :]
        y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = torchF.conv1d(xpadded, torch.flip(self.conv_weights, (0,))[None, None, :],
                                                                                      stride=self.stride).squeeze_()

        return y

    def forward_numpy(self, x: torch.TensorType):
        xnp = x.numpy()
        y = np.convolve(np.pad(xnp, self.padding, mode='constant', constant_values=0.0),
                        self.conv_weights.numpy(), mode='valid')[::self.stride]
        return torch.from_numpy(y)

    def get_filter(self):
        #return self.conv.weight.squeeze().detach().cpu().numpy()
        return self.conv_weights.detach().cpu().numpy()

    def normalize_filter(self):
        # Calculate L2 norm and divide on the filter
        with torch.no_grad():
            self.conv_weights /= torch.linalg.norm(self.conv_weights)
    
    def set_stride(self, new_stride):
        self.stride = new_stride


class MultiChannelFIRfilter(FIRfilter):
    """
        Applies same FIR filter to n channels
    """
    def forward(self, x):
        # input x assumed to be [timesteps, n_channels]
        # xpadded = torch.permute(torchF.pad(x, self.padding, mode='constant', value=0.0), (1,0))[None, :, :]
        f_out = torchF.conv1d(torch.permute(x, (1,0)), torch.flip(self.conv_weights, (0,))[None, None, :].repeat(x.shape[1], 1, 1),
                              stride=self.stride, padding=self.padding[0],
                              groups=x.shape[1]).squeeze_()
        return torch.permute(f_out, (1, 0))

    def forward_batched(self, x, batch_size=2000):
        raise NotImplementedError

    def forward_numpy(self, x: torch.TensorType):
        xnp = x.numpy()
        y = np.empty((len(xnp) // self.stride, x.shape[0]))
        for i in range(x.shape[1]):
            y[:, i] = np.convolve(np.pad(xnp[:, i], self.padding, mode='constant', constant_values=0.0),
                                self.conv_weights.numpy(), mode='valid')[::self.stride]
        return torch.from_numpy(y)


class BesselFilter(torch.nn.Module):
    """
        Bessel filter as a torch module. No learnable parameters.
    """
    def __init__(self, bessel_order: int, cutoff_hz: float, fs: float,
                 dtype=torch.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        bessel_b, bessel_a = bessel(bessel_order, cutoff_hz, fs=fs, norm='mag')
        self.filter_b, self.filter_a = torch.from_numpy(bessel_b), torch.from_numpy(bessel_a)
        # Crude estimate of sample delay through filter - take average in passband
        f, gd = group_delay((bessel_b, bessel_a), fs=fs)
        self.delay = np.average(gd[np.where(f < cutoff_hz)])

    def forward(self, x):
        # lfilter assumes input is between -1 and 1. Do so and rescale afterwards.
        xmax = torch.max(torch.abs(x))
        return xmax *  lfilter(x / xmax, self.filter_a, self.filter_b)
    
    def forward_numpy(self, y):
        # convert to numpy - uses scipys lfilter and coverts back
        # only to be used during eval
        return torch.from_numpy(scipy_lfilter(self.filter_b.numpy(),
                                              self.filter_a.numpy(),
                                              y.numpy()))

    def forward_batched(self, x, batch_size):
        print("BesselFilter: Warning! Using Numpy forward method (ignoring batch size)")
        return self.forward_numpy(x)

    def get_filters(self):
        return self.filter_b.detach().cpu().numpy(), self.filter_a.detach().cpu().numpy()
    
    def get_sample_delay(self):
        return self.delay


class MultiChannelBesselFilter(BesselFilter):
    """
        Bessel filter as a torch module. No learnable parameters.
    """
    def forward(self, x):
        # lfilter assumes input is between -1 and 1. Do so and rescale afterwards.
        xmax = torch.max(torch.abs(x))
        return torch.permute(xmax *  lfilter(torch.permute(x, (1, 0)) / xmax, self.filter_a, self.filter_b), (1,0))
    
    def forward_numpy(self, y):
        # convert to numpy - uses scipys lfilter and coverts back
        # only to be used during eval
        ynp = y.numpy()
        y_out = np.zeros_like(ynp)
        for i in range(ynp.shape[1]):
            y_out[:, i] = scipy_lfilter(self.filter_b.numpy(),
                                        self.filter_a.numpy(),
                                        ynp[:, i])
        return torch.from_numpy(y_out)


class BrickWallFilter(torch.nn.Module):
    """
        Ideal sinc filter for low pass filtering. No learnable parameters.
    """
    def __init__(self, filter_length: int, cutoff_hz: float, fs: float,
                 dtype=torch.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ti = np.arange(-filter_length // 2, filter_length // 2) * 1/fs
        pulse = 2 * cutoff_hz * np.sinc(2 * cutoff_hz * ti)
        self.filter = torch.from_numpy(pulse / np.linalg.norm(pulse))
        self.padding = ((filter_length - 1) // 2, (filter_length - filter_length % 2) // 2)

    def forward(self, x):
        xpadded = torchF.pad(x, self.padding, mode='constant', value=0.0)[None, None, :]
        f_out = torchF.conv1d(xpadded, torch.flip(self.filter, (0,))[None, None, :]).squeeze_()
        return f_out

    def forward_batched(self, x, batch_size=2000):
        # Forward pass that uses batching but without boundary effects bewteen batches
        assert batch_size > self.padding[0]
        # Allocate output
        y = torch.empty((x.shape[0],), dtype=x.dtype)
        n_batches = int(np.ceil(len(x) / batch_size))
        prepad = torch.zeros((self.padding[0]))
        postpad = x[batch_size:(batch_size+self.padding[1])]
        outputs_pr_batch = batch_size

        for b in range(n_batches - 2):
            xpadded = torch.concat((prepad,
                                    x[b*batch_size:(b*batch_size + batch_size)],
                                    postpad))[None, None, :]
            y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = torchF.conv1d(xpadded, torch.flip(self.filter, (0,))[None, None, :]).squeeze_()

            # Update prepad and postpad
            prepad = x[(b+1)*batch_size-self.padding[0]:(b+1)*batch_size]
            postpad = x[((b + 2)*batch_size):((b + 2)*batch_size + self.padding[1])]

        # Calcualte second to last batch
        b = n_batches - 2
        xpadded = torch.concat((prepad,
                                x[b*batch_size:(b*batch_size + batch_size)],
                                postpad))[None, None, :]
        y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = torchF.conv1d(xpadded, torch.flip(self.filter, (0,))[None, None, :]).squeeze_()

        # For the last batch postpad with zeros
        prepad = x[(b+1)*batch_size-self.padding[0]:(b+1)*batch_size]
        b = n_batches - 1
        xpadded = torch.concat((prepad,
                                x[b*batch_size:(b*batch_size + batch_size)],
                                torch.zeros((self.padding[1]))))[None, None, :]
        y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = torchF.conv1d(xpadded, torch.flip(self.filter, (0,))[None, None, :]).squeeze_()

        return y

    def get_filters(self):
        return self.filter.detach().cpu().numpy(), 1


class GaussianFqFilter(torch.nn.Module):
    def __init__(self, filter_cutoff_hz, order, Fs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Fs = Fs
        self.order = order
        self.filter_cutoff_hz = filter_cutoff_hz
        self.logsqrt2 = np.log(np.sqrt(2))

    def _calculate_filter(self, f):
        return torch.exp(-self.logsqrt2*(2*f/(2*self.filter_cutoff_hz))**(2 * self.order))

    def forward(self, x):  
        fqs = fftshift(fftfreq(len(x), 1/self.Fs))
        filter = self._calculate_filter(fqs)
        X = fftshift(fft(x))/len(x)
        return ifft(ifftshift(X * filter))*len(X)
    
    def forward_batched(self, x, batch_size=None):
        return self.forward_numpy(x)

    def forward_numpy(self, x):
        # Only use for evaluation. Will not track gradients
        xnp = x.numpy()
        fqs = np.linspace(-0.5, 0.5, num=len(xnp)) * self.Fs
        filter = self._calculate_filter(torch.from_numpy(fqs)).numpy()
        X = np.fft.fftshift(np.fft.fft(xnp))/len(xnp)
        return torch.from_numpy(np.fft.ifft(np.fft.ifftshift(X * filter))*len(X))
