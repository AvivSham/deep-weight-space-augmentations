from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class WeightAndBiases:
    # todo: this should be initialized from sd so we need to save weights and biases names to be
    #  able to load them later with ease
    weights: Tuple[torch.Tensor, ...]
    biases: Tuple[torch.Tensor, ...]


@dataclass
class MixUpOutput:
    weights: Tuple[torch.Tensor, ...]
    biases: Tuple[torch.Tensor, ...]
    alpha: torch.Tensor
