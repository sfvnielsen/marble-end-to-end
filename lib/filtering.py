import torch
from torch.nn import functional as torchF
from torchaudio.functional import convolve, lfilter
import numpy.typing as npt
import numpy as np
from scipy.signal import bessel

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
    
    def get_filters(self):
        return None, None


class FIRfilter(torch.nn.Module):
    def __init__(self, filter_weights: npt.ArrayLike, stride=1, normalize=False, trainable=False, dtype=torch.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        torch_filter = torch.from_numpy(np.copy(filter_weights))
        self.filter_length = len(filter_weights)
        self.padding = ((self.filter_length - 1) // 2, (self.filter_length - self.filter_length % 2) // 2)
        self.conv_weights = torch.nn.Parameter(torch_filter, requires_grad=trainable)
        self.normalize = normalize
        self.trainalbe = trainable
        self.stride = stride

    def forward(self, x):
        # input x assumed to be [timesteps,]
        xpadded = torchF.pad(x, self.padding, mode='constant', value=0.0)[None, None, :]
        f_out = torchF.conv1d(xpadded, torch.flip(self.conv_weights, (0,))[None, None, :],
                              stride=self.stride).squeeze_()
        if self.normalize:
            #f_out = f_out.div_(torch.sqrt(torch.sum(torch.square(self.conv.weight))))
            f_out = f_out.div_(torch.sqrt(torch.sum(torch.square(self.conv_weights))))
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

    def get_filter(self):
        #return self.conv.weight.squeeze().detach().cpu().numpy()
        return self.conv_weights.detach().cpu().numpy()

    def normalize_filter(self):
        # Calculate L2 norm and divide on the filter
        with torch.no_grad():
            self.conv_weights /= torch.linalg.norm(self.conv_weights)


class BesselFilter(torch.nn.Module):
    """
        Bessel filter as a torch module. No learnable parameters.
    """
    def __init__(self, bessel_order: int, cutoff_hz: float, fs: float,
                 dtype=torch.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        bessel_b, bessel_a = bessel(bessel_order, cutoff_hz, fs=fs, norm='mag')
        self.filter_b, self.filter_a = torch.from_numpy(bessel_b), torch.from_numpy(bessel_a)

    def forward(self, x):
        # lfilter assumes input is between -1 and 1. Do so and rescale afterwards.
        xmax = torch.max(torch.abs(x))
        return xmax *  lfilter(x / xmax, self.filter_a, self.filter_b)

    def forward_batched(self, x, batch_size):
        print("BesselFilter: Warning! 'forward_batched' method not implemented yet. Deferring to forward.")
        return self.forward(x)

    def get_filters(self):
        return self.filter_b.detach().cpu().numpy(), self.filter_a.detach().cpu().numpy()


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
