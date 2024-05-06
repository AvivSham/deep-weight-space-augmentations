import math
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight: torch.Tensor, bias: torch.Tensor, c: float, w0: float):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            # bias.uniform_(-w_std, w_std)
            bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class INR(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        n_layers: int = 3,
        up_scale: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        hidden_dim = in_dim * up_scale

        self.layers = [Siren(dim_in=in_dim, dim_out=hidden_dim)]
        for i in range(n_layers - 2):
            self.layers.append(Siren(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_channels))
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + 0.5


class FunctionalSiren(nn.Module):
    def __init__(
        self,
        w0=30.0,
        c=6.0,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.activation = Sine(w0) if activation is None else activation

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        out = F.linear(x, weight, bias)
        out = self.activation(out)
        return out


class FunctionalINR(nn.Module):
    def __init__(
        self, in_dim: int = 2
    ):
        super().__init__()

        self.f_siren = FunctionalSiren()

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor
    ) -> torch.Tensor:
        for w, b in zip(weights[:-1], biases[:-1]):
            x = self.f_siren(
                x, w.squeeze(0).squeeze(-1).transpose(1, 0), b.squeeze(0).squeeze(-1)
            )
        x = F.linear(
            x,
            weights[-1].squeeze(0).squeeze(-1).transpose(1, 0),
            biases[-1].squeeze(0).squeeze(-1),
        )
        return x + 0.5
