"""
    Library of equalizer structures - to be used inside a system
    (cf. systems.py)
"""

import numpy as np
import torch

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

        self.filter = FIRfilter(filter_weights=weights, stride=self.samples_per_symbol, normalize=False,
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