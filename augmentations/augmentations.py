import copy
import random
import time
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from experiments.utils.utils import make_coordinates
from experiments.utils.data_objects import WeightAndBiases, MixUpOutput
from experiments.utils.matching.git_rebasin import merge_two


def extract_weights_and_biases(sd):
    weights = []
    biases = []
    for k, v in sd.items():
        if "weight" in k:
            if len(v.shape) < 3:
                v = v.permute(1, 0).unsqueeze(0).unsqueeze(-1)
            weights.append(v)
        elif "bias" in k:
            if len(v.shape) < 2:
                v = v.unsqueeze(0).unsqueeze(-1)
            biases.append(v)
        else:
            raise ValueError(
                f"{k} is not supported should contain weight or bias in the name"
            )
    return tuple(weights), tuple(biases)


class BaseAugmentation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, weights_and_biases: WeightAndBiases, *args, **kwargs) -> WeightAndBiases:
        raise NotImplementedError


class BaseMixUpAugmentation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(
        self,
        w_and_b_0: Tuple[torch.Tensor],
        w_and_b_1: Tuple[torch.Tensor],
        label_0: Union[Tuple[torch.Tensor], int],
        label_1: Union[Tuple[torch.Tensor], int],
    ) -> MixUpOutput:
        raise NotImplementedError

    def _merge_label(self, label_0, label_1, alpha):
        if isinstance(label_0, int):
            return alpha * label_0 + (1 - alpha) * label_1
        elif isinstance(label_0, tuple):
            weights_0, biases_0 = label_0
            weights_1, biases_1 = label_1
            return self._avg_weights_and_biases(
                weights_0, biases_0, weights_1, biases_1
            )
        elif isinstance(label_0, torch.Tensor):
            return alpha * label_0 + (1 - alpha) * label_1
        else:
            raise TypeError(
                f"labels are type: {type(label_0)} only [int, tuple] are supported"
            )

    @staticmethod
    def _avg_weights_and_biases(weights0, biases0, weights1, biases1, alpha=0.5):
        avg_weights = tuple(
            alpha * w0 + (1 - alpha) * w1 for w0, w1 in zip(weights0, weights1)
        )
        avg_bias = tuple(
            alpha * b0 + (1 - alpha) * b1 for b0, b1 in zip(biases0, biases1)
        )
        return avg_weights, avg_bias


class LabelSmoothingAugmentation(BaseAugmentation):
    def __init__(self, smoothing_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing_rate = smoothing_rate

    def forward(self, weights_and_biases: WeightAndBiases, label, num_classes) -> Tuple[WeightAndBiases, torch.Tensor]:
        label = (
                (1 - self.smoothing_rate) * F.one_hot(torch.tensor(label), num_classes) +
                (self.smoothing_rate / num_classes)
        )
        return weights_and_biases, label


class DropOutAugmentation(BaseAugmentation):
    def __init__(self, drop_rate: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]
        return WeightAndBiases(tuple(new_weights), tuple(new_biases))


class TranslateAugmentation(BaseAugmentation):
    """NOTE: 1. for INRs. 2. not exactly translate when we propagate to deeper layers"""

    def __init__(self, translation_scale: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.translation_scale = translation_scale

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )
        translation_shape = new_weights[0].shape[:2] + new_weights[0].shape[3:]
        translation = torch.empty(translation_shape).uniform_(
            -self.translation_scale, self.translation_scale
        )
        # NOTE: Not exactly translate when we propagate to deeper layers
        order = random.sample(range(1, len(new_weights)), 1)[0]
        bias_res = translation
        i = 0
        for i in range(order):
            bias_res = torch.einsum("bic,bijc->bjc", bias_res, new_weights[i])

        new_biases[i] += bias_res
        return WeightAndBiases(tuple(new_weights), tuple(new_biases))


class RotateAugmentation(BaseAugmentation):
    """NOTE: for 2D INRs."""

    def __init__(self, rotation_degree: float = 30., **kwargs):
        super().__init__(**kwargs)
        self.rotation_degree = rotation_degree

    @staticmethod
    def rotation_mat(degree=30.0):
        angle = torch.empty(1).uniform_(-degree, degree)
        angle_rad = angle * (torch.pi / 180)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        return rotation_matrix

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )

        assert (
            new_weights[0].shape[1] == 2
        ), "Only 2D INRs are supported for RotateAugmentation"
        rot_mat = self.rotation_mat(self.rotation_degree)
        new_weights[0] = torch.einsum("ij,bjkc->bikc", rot_mat, new_weights[0])

        return WeightAndBiases(tuple(new_weights), tuple(new_biases))


class GaussianNoiseAugmentation(BaseAugmentation):
    """NOTE: noise scale is multiplied by the weights std"""

    def __init__(self, noise_scale: float = 0.32, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )
        new_weights = [
            w
            + torch.empty(w.shape).normal_(
                0, self.noise_scale * (torch.nan_to_num(w.std()) + 1e-8)
            )
            for w in new_weights
        ]
        new_biases = [
            b
            + torch.empty(b.shape).normal_(
                0, self.noise_scale * (torch.nan_to_num(b.std()) + 1e-8)
            )
            for b in new_biases
        ]
        return WeightAndBiases(tuple(new_weights), tuple(new_biases))


class ScaleAugmentation(BaseAugmentation):
    def __init__(self, resize_scale: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.resize_scale = resize_scale

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
        new_weights[0] = new_weights[0] * rand_scale
        return WeightAndBiases(tuple(new_weights), tuple(new_biases))


class QuantileAugmentation(BaseAugmentation):
    def __init__(self, quantile: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert quantile >= 0, "quantile must be >= 0"
        self.quantile = quantile

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        if self.quantile > 0:
            new_weights, new_biases = list(weights_and_biases.weights), list(
                weights_and_biases.biases
            )
            do_q = torch.empty(1).uniform_(0, self.quantile)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]
            return WeightAndBiases(tuple(new_weights), tuple(new_biases))

        return weights_and_biases
    

class SirenNegationAugmentation(BaseAugmentation):
    def __init__(self, negation_prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.negation_prob = negation_prob

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )
        for i in range(len(new_weights) - 1):  # final layer not Siren
            negate = torch.rand_like(new_biases[i]) < self.negation_prob
            sign = negate.float() * -2 + 1
            new_weights[i] = sign.unsqueeze(1) * new_weights[i]
            new_biases[i] = sign * new_biases[i]
            new_weights[i + 1] = sign.unsqueeze(2) * new_weights[i + 1]

        return WeightAndBiases(tuple(new_weights), tuple(new_biases))
    

class SirenBiasAugmentation(BaseAugmentation):
    def __init__(self, k_max: int = 4, w0: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.k_max = k_max
        self.w0 = w0

    def forward(self, weights_and_biases: WeightAndBiases) -> WeightAndBiases:
        new_weights, new_biases = list(weights_and_biases.weights), list(
            weights_and_biases.biases
        )
        for i in range(len(new_weights) - 1):  # final layer not Siren
            k = torch.randint_like(new_biases[i], -self.k_max, self.k_max)
            new_biases[i] = new_biases[i] + (k * torch.pi) / self.w0
            negate = k % 2 == 1
            sign = negate.float() * -2 + 1
            new_weights[i + 1] = sign.unsqueeze(2) * new_weights[i + 1]

        return WeightAndBiases(tuple(new_weights), tuple(new_biases))


class VanillaMixUp(BaseMixUpAugmentation):
    def forward(self, weights, biases, weights_rand, biases_rand, **kwargs) -> MixUpOutput:
        alpha = torch.rand(1)
        weights_and_biases = self._avg_weights_and_biases(
            weights,
            biases,
            weights_rand,
            biases_rand,
            alpha
        )
        return MixUpOutput(*weights_and_biases, alpha)


class OnlyWeightsAndBiasesMixUp(BaseMixUpAugmentation):
    """Perform Mixup only on the weights and biases and not on the labels

    """
    def forward(self, weights, biases, weights_rand, biases_rand, **kwargs) -> MixUpOutput:
        alpha = torch.rand(1)
        weights_and_biases = self._avg_weights_and_biases(
            weights,
            biases,
            weights_rand,
            biases_rand,
            alpha
        )
        # NOTE: we set alpha to 1 or 0 according to whether alpha >= 0.5
        return MixUpOutput(*weights_and_biases, (alpha >= 0.5).float())


class MixUpWithPerm(BaseMixUpAugmentation):
    def forward(self, weights, biases, weights_rand, biases_rand, **kwargs) -> MixUpOutput:
        w_and_b_0 = (weights, biases)
        w_and_b_1 = (weights_rand, biases_rand)
        dim = len(w_and_b_0[0][0].shape)
        if (w_and_b_0[0][0].shape[0] == 1) and (dim > 2):
            weights = tuple([w.squeeze(0) for w in w_and_b_0[0]])
            biases = tuple([b.squeeze(0) for b in w_and_b_0[1]])
        else:
            weights, biases = w_and_b_0
        perm_w_and_b_0 = self._permute(weights, biases)
        alpha = torch.rand(1)
        weights_and_biases = self._avg_weights_and_biases(*perm_w_and_b_0, *w_and_b_1, alpha)
        return MixUpOutput(*weights_and_biases, alpha)

    @staticmethod
    def _permute(weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        return tuple(new_weights), tuple(new_biases)


class ReBasinMixUpAugmentation(BaseMixUpAugmentation):
    def __init__(self, low_bound=0., up_bound=1., **kwargs):
        super().__init__(**kwargs)
        self.low_bound = low_bound
        self.up_bound = up_bound

    def forward(
        self,
        weights: Tuple[torch.Tensor],
        biases: Tuple[torch.Tensor],
        weights_rand: Tuple[torch.Tensor],
        biases_rand: Tuple[torch.Tensor],
        weight_names,
        bias_names,
    ) -> MixUpOutput:
        alpha = torch.rand(1).item() * (self.up_bound - self.low_bound) + self.low_bound
        # alignment
        model_a_sd, model_b_sd = {}, {}
        # NOTE: we do this to retain the order of the original model
        for i in range(len(weights)):
            model_a_sd[weight_names[i]] = weights[i].squeeze(0).squeeze(-1).permute(1, 0)
            model_a_sd[bias_names[i]] = biases[i].squeeze(0).squeeze(-1)
            model_b_sd[weight_names[i]] = weights_rand[i].squeeze(0).squeeze(-1).permute(1, 0)
            model_b_sd[bias_names[i]] = biases_rand[i].squeeze(0).squeeze(-1)

        # NOTE: this modifies model_b_sd inplace
        merge_two(model_a=model_a_sd, models=[model_b_sd], num_hidden_layers=len(weights) - 1)
        weights_rand = tuple([v.permute(1, 0).unsqueeze(-1) for w, v in model_b_sd.items() if "weight" in w])
        biases_rand = tuple([v.unsqueeze(-1) for w, v in model_b_sd.items() if "bias" in w])

        weights = tuple(
            (1 - alpha) * weights_rand[i] + alpha * weights[i] for i in range(len(weights)))
        biases = tuple(
            (1 - alpha) * biases_rand[i] + alpha * biases[i] for i in range(len(biases)))

        return MixUpOutput(weights=weights, biases=biases, alpha=alpha)


class CombinationAugmentation(ReBasinMixUpAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        augmentations = [
            TranslateAugmentation(),
            GaussianNoiseAugmentation(),
            SirenNegationAugmentation(),
        ]
        self.augmentations = nn.Sequential(*augmentations)
        self.mixup_aug = ReBasinMixUpAugmentation()

    def forward(
            self,
            weights: Tuple[torch.Tensor],
            biases: Tuple[torch.Tensor],
            weights_rand: Tuple[torch.Tensor],
            biases_rand: Tuple[torch.Tensor],
            weight_names,
            bias_names,
    ) -> MixUpOutput:
        weights_and_biases = self.augmentations(WeightAndBiases(weights, biases))
        weights, biases = weights_and_biases.weights, weights_and_biases.biases
        return self.mixup_aug(weights, biases, weights_rand, biases_rand, weight_names, bias_names)


class SelfRebasinMixUpAugmentation(ReBasinMixUpAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.augmentations = [v for k, v in name2aug.items() if k != "label_smoothing"]
        self.mixup_aug = ReBasinMixUpAugmentation()

    def forward(
            self,
            weights: Tuple[torch.Tensor],
            biases: Tuple[torch.Tensor],
            **kwargs
    ) -> MixUpOutput:
        weight_names, bias_names = kwargs["weight_names"], kwargs["bias_names"]
        aug = random.choice(self.augmentations)
        orig_weights, orig_biases = copy.deepcopy(weights), copy.deepcopy(biases)
        augmented_weights_and_biases = aug(WeightAndBiases(weights, biases))
        aug_weights, aug_biases = augmented_weights_and_biases.weights, augmented_weights_and_biases.biases
        return self.mixup_aug(orig_weights, orig_biases, aug_weights, aug_biases, weight_names, bias_names)


name2aug = {
    "dropout": DropOutAugmentation(),
    "translate": TranslateAugmentation(),
    "rotate": RotateAugmentation(),
    "gaussian_noise": GaussianNoiseAugmentation(),
    "scale": ScaleAugmentation(),
    "quantile": QuantileAugmentation(),
    "siren_negation": SirenNegationAugmentation(),
    "siren_bias": SirenBiasAugmentation(),
    "label_smoothing": LabelSmoothingAugmentation(),
}

mixup_augs = {
    "self_rebasin": SelfRebasinMixUpAugmentation(),
    "rebasin": ReBasinMixUpAugmentation(),
    "mixup": VanillaMixUp(),
    "mixup_and_perm": MixUpWithPerm(),
    "combine": CombinationAugmentation(),
    "weights_only_aug": OnlyWeightsAndBiasesMixUp(),
}
