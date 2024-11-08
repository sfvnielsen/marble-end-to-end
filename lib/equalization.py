"""
    Library of equalizer structures - to be used inside a system
    (cf. systems.py)
"""

import numpy as np
import torch
import torch.nn.functional as F

from .filtering import FIRfilter


class GenericTorchEqualizer(object):
    """ Parent class that implements a general equalizer structure
        Parameters of the equaliser are optimized by the consuming class
        through the `get_parameters` interface
    """
    def __init__(self, samples_per_symbol, dtype=torch.float32, torch_device=torch.device("cpu")) -> None:
        self.samples_per_symbol = samples_per_symbol
        self.torch_device = torch_device
        self.dtype = dtype

    # weak implementation
    def forward(self, y):
        raise NotImplementedError
    
    # weak implementation
    def forward_batched(self, y, batch_size):
        raise NotImplementedError
    
    # weak implementation
    def zero_grad(self) -> None:
        raise NotImplementedError

    # weak implementation
    def get_parameters(self):
        raise NotImplementedError

    # weak implementation
    def print_model_parameters(self):
        raise NotImplementedError

    # weak implementation
    def train_mode(self):
        raise NotImplementedError

    # weak implementation
    def eval_mode(self):
        raise NotImplementedError

    # weak implementation
    def __repr__(self) -> str:
        raise NotImplementedError
    

class LinearFeedForwardEqualiser(GenericTorchEqualizer):
    """
        Vanilla FFE. Uses conv1d as main operation
    """
    def __init__(self, n_taps,
                 samples_per_symbol, dtype=torch.float32, torch_device=torch.device("cpu")) -> None:
        super().__init__(samples_per_symbol, dtype, torch_device)

        # Intialize equaliser filter
        assert (n_taps + 1) % 2 == 0
        weights = np.zeros((n_taps, ))
        weights[n_taps // 2] = 1.0

        self.filter = FIRfilter(filter_weights=weights, stride=self.samples_per_symbol,
                                   trainable=True, dtype=dtype)

    def get_parameters(self):
        return [{'params': self.filter.conv_weights}]

    def set_stride(self, new_stride):
        self.filter.set_stride(new_stride)
    
    def forward(self, y):
        y_eq = self.filter.forward(y)
        return y_eq
    
    def forward_batched(self, y, batch_size=None):
        return self.filter.forward_numpy(y)
    
    def zero_grad(self):
        self.filter.zero_grad()
    
    def print_model_parameters(self):
        print(self.filter.weight)

    def train_mode(self):
        self.filter.train()

    def eval_mode(self):
        self.filter.eval()

    def __repr__(self) -> str:
        return f"LinearFFE({len(self.filter.weight)})"


class SecondOrderVolterraSeries(torch.nn.Module):
    """
        Vanilla second order Volterra-series implemented in torch using einsum
    """
    def __init__(self, n_lags1, n_lags2, samples_per_symbol, dtype=torch.double,
                 torch_device=torch.device('cpu'),
                 **kwargs) -> None:
        super(SecondOrderVolterraSeries, self).__init__(**kwargs)

        self.sps = samples_per_symbol
        self.dtype = dtype
        self.torch_device = torch_device
        self.stride = samples_per_symbol
        assert (n_lags1 + 1) % 2 == 0  # assert odd lengths
        assert (n_lags2 + 1) % 2 == 0

        # Initialize first order kernel
        kernel1_init = np.zeros((n_lags1, ))
        kernel1_init[n_lags1 // 2] = 1.0  # dirac initialization
        self.kernel1 = torch.nn.Parameter(torch.from_numpy(kernel1_init).to(dtype).to(self.torch_device), requires_grad=True)
        self.lag1_padding = n_lags1 // 2

        # Initialize second order kernel (zeros)
        kernel2_init = torch.zeros((n_lags2, n_lags2), dtype=dtype)
        self.kernel2 = torch.nn.Parameter(kernel2_init.to(self.torch_device))
        self.lag2_padding = n_lags2 // 2

    def forward(self, x: torch.TensorType):
        # Output of first order kernel
        xpad = torch.concatenate((torch.zeros((self.lag1_padding), dtype=self.dtype, device=self.torch_device),
                                  x,
                                  torch.zeros((self.lag1_padding), dtype=self.dtype, device=self.torch_device)))
        y = F.conv1d(xpad[None, None, :], self.kernel1[None, None, :], stride=self.stride).squeeze()

        # Create lag and calculate the pr. lag outer product
        x2pad = torch.concatenate((torch.zeros((self.lag2_padding), dtype=self.dtype, device=self.torch_device),
                                   x,
                                   torch.zeros((self.lag2_padding), dtype=self.dtype, device=self.torch_device)))
        Xlag = torch.flip(x2pad.unfold(0, self.kernel2.shape[0], self.stride), (1,))
        Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)

        # Apply kernel to outer prodcts pr lag - symmetrize kernel
        y2 = torch.einsum('ijk,jk->i', Xouter, self.kernel2 + self.kernel2.T)

        return y + y2
    
    def forward_batched(self, x: torch.TensorType, batch_size=2000):
        # Forward pass that uses batching but without boundary effects bewteen batches
        assert batch_size > self.lag1_padding
        assert batch_size > self.lag2_padding
        assert batch_size % self.stride == 0

        # Allocate output
        y = torch.empty((x.shape[0] // self.stride,), dtype=x.dtype)
        n_batches = int(np.ceil(len(x) / batch_size))
        prepad1 = torch.zeros((self.lag1_padding))
        postpad1 = x[batch_size:(batch_size+self.lag1_padding)]
        prepad2 = torch.zeros((self.lag2_padding))
        postpad2 = x[batch_size:(batch_size+self.lag2_padding)]
        outputs_pr_batch = batch_size // self.stride

        for b in range(n_batches - 2):
            # Output of first order kernel
            xpadded = torch.concat((prepad1,
                                    x[b*batch_size:(b*batch_size + batch_size)],
                                    postpad1))[None, None, :]
            this_y = F.conv1d(xpadded,
                              self.kernel1[None, None, :], stride=self.stride).squeeze()

            # Output of second order kernel
            x2pad = torch.concatenate((prepad2,
                                       x[b*batch_size:(b*batch_size + batch_size)],
                                       postpad2))
            Xlag = torch.flip(x2pad.unfold(0, self.kernel2.shape[0], self.stride), (1,))
            Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)
            this_y2 = torch.einsum('ijk,jk->i', Xouter, self.kernel2 + self.kernel2.T)

            # Update prepad and postpad
            prepad1 = x[(b+1)*batch_size-self.lag1_padding:(b+1)*batch_size]
            postpad1 = x[((b + 2)*batch_size):((b + 2)*batch_size + self.lag1_padding)]
            prepad2 = x[(b+1)*batch_size-self.lag2_padding:(b+1)*batch_size]
            postpad2 = x[((b + 2)*batch_size):((b + 2)*batch_size + self.lag2_padding)]

            # Update output vector
            y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = this_y + this_y2

        # Calcualte second to last batch
        b = n_batches - 2
        xpadded = torch.concat((prepad1,
                                x[b*batch_size:(b*batch_size + batch_size)],
                                postpad1))[None, None, :]
        this_y = F.conv1d(xpadded,
                            self.kernel1[None, None, :], stride=self.stride).squeeze()

        # Output of second order kernel
        x2pad = torch.concatenate((prepad2,
                                   x[b*batch_size:(b*batch_size + batch_size)],
                                   postpad2))
        Xlag = torch.flip(x2pad.unfold(0, self.kernel2.shape[0], self.stride), (1,))
        Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)
        this_y2 = torch.einsum('ijk,jk->i', Xouter, self.kernel2 + self.kernel2.T)
        y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = this_y + this_y2
        
        # For the last batch postpad with zeros
        prepad1 = x[(b+1)*batch_size-self.lag1_padding:(b+1)*batch_size]
        postpad1 = torch.zeros((self.lag1_padding))
        prepad2 = x[(b+1)*batch_size-self.lag2_padding:(b+1)*batch_size]
        postpad2 = torch.zeros((self.lag2_padding))
        b = n_batches - 1
        xpadded = torch.concat((prepad1,
                                x[b*batch_size:(b*batch_size + batch_size)],
                                postpad1))[None, None, :]
        this_y = F.conv1d(xpadded,
                          self.kernel1[None, None, :], stride=self.stride).squeeze()

        # Output of second order kernel
        x2pad = torch.concatenate((prepad2,
                                   x[b*batch_size:(b*batch_size + batch_size)],
                                   postpad2))
        Xlag = torch.flip(x2pad.unfold(0, self.kernel2.shape[0], self.stride), (1,))
        Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)
        this_y2 = torch.einsum('ijk,jk->i', Xouter, self.kernel2 + self.kernel2.T)
        y[b*outputs_pr_batch:(b*outputs_pr_batch + outputs_pr_batch)] = this_y + this_y2

        return y

    
    def set_stride(self, new_stride: int) -> None:
        self.stride = new_stride


class VolterraEqualizer(GenericTorchEqualizer):
    """
        Volterra equalizer with a second order kernel.
        Uses the SecondOrderVolterraSeries module above.
    """
    def __init__(self, samples_per_symbol, n_lags1: int, n_lags2: int,
                 dtype=torch.float32, torch_device=torch.device("cpu")) -> None:
        super().__init__(samples_per_symbol, dtype, torch_device)
        self.equalizer = SecondOrderVolterraSeries(n_lags1=n_lags1, n_lags2=n_lags2, samples_per_symbol=samples_per_symbol,
                                                   dtype=dtype, torch_device=torch_device)

   # weak implementation
    def forward(self, y):
        return self.equalizer.forward(y)
    
    # weak implementation
    def forward_batched(self, y, batch_size):
        return self.equalizer.forward_batched(y, batch_size)
    
    def zero_grad(self) -> None:
        self.equalizer.zero_grad()

    def get_parameters(self):
        return [{"params": self.equalizer.parameters()}]

    # weak implementation
    def print_model_parameters(self):
        raise NotImplementedError

    def train_mode(self):
        self.equalizer.train()

    def eval_mode(self):
        self.equalizer.eval()
    
    def set_stride(self, new_stride: int) -> None:
        self.equalizer.set_stride(new_stride)

    def __repr__(self) -> str:
        return f"VNLE({len(self.equalizer.kernel1)}, {self.equalizer.kernel2.shape[0]})"