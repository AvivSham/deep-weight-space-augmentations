import random
import json
from functools import partial
from pathlib import Path
from typing import NamedTuple, Tuple, Union, List

import torch
import torch.nn.functional as F

from experiments.utils.data_objects import WeightAndBiases
from augmentations.augmentations import (
    BaseAugmentation,
    BaseMixUpAugmentation,
    LabelSmoothingAugmentation
)


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class INRDataset(torch.utils.data.Dataset):
    """Base Dataset class for INR classification/regression"""

    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        do_augmentation=False,
        permutation=False,
        statistics_path=None,
        augmentations: List[BaseAugmentation] = None,
        mixup_augmentation: BaseMixUpAugmentation = None,
        num_classes=10,
    ):
        self.split = split

        # handling case that paths in split file and splits location are not identical.
        # we assume that splits.json is located inside destination data dir.
        root_path = Path(path).parent
        self.dataset = json.load(open(path, "r"))[self.split]
        self.dataset = list(map(partial(self._change_root_path, root_p=root_path), self.dataset))

        self.do_augmentation = do_augmentation
        self.permutation = permutation
        self.normalize = normalize
        if self.normalize:
            assert statistics_path is not None
            self.stats = torch.load(statistics_path, map_location="cpu")

        augmentations = augmentations if augmentations is not None else []
        self.augmentations = torch.nn.Sequential(*augmentations)
        self.mixup_augmentation = mixup_augmentation
        self.num_classes = (
            torch.tensor(num_classes) if num_classes is not None else None
        )

    @staticmethod
    def _change_root_path(orig_p, root_p):
        orig_p = Path(orig_p)
        d_name = root_p.name
        orig_parts = list(orig_p.parts)
        sample_path_from_parent = "/".join(orig_parts[orig_parts.index(d_name) + 1:])
        return (root_p / sample_path_from_parent).as_posix()

    def __len__(self):
        return len(self.dataset)

    def _load_sd_as_weights_biases_label(self, item, return_names=False):
        path = self.dataset[item]
        state_dict = torch.load(path, map_location="cpu")
        label = state_dict.pop("label")
        weights = tuple(
            [
                v.permute(1, 0).unsqueeze(0).unsqueeze(-1)
                for w, v in state_dict.items()
                if "weight" in w
            ]
        )
        biases = tuple(
            [v.unsqueeze(0).unsqueeze(-1) for w, v in state_dict.items() if "bias" in w]
        )
        if return_names:
            weight_names = [k for k in state_dict.keys() if "weight" in k]
            bias_names = [k for k in state_dict.keys() if "bias" in k]
            return weights, biases, label, weight_names, bias_names
        else:
            return weights, biases, label

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    def _augment(self, weights, biases, label):
        """translation and rotation

        :param weights:
        :param biases:
        :return:
        """
        weight_and_biases = WeightAndBiases(weights, biases)
        if len(self.augmentations) and isinstance(self.augmentations[0], LabelSmoothingAugmentation):
            augmented_weights_and_biases, label = self.augmentations[0](
                weights_and_biases=weight_and_biases, label=label, num_classes=self.num_classes
            )
        else:
            augmented_weights_and_biases = self.augmentations(weight_and_biases)

        new_weights, new_biases = (
            augmented_weights_and_biases.weights,
            augmented_weights_and_biases.biases,
        )
        return new_weights, new_biases, label

    @staticmethod
    def _permute(weights, biases, return_permutation=False):
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
        if return_permutation:
            return tuple(new_weights), tuple(new_biases), tuple(perms)
        else:
            return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _align_dims(weights, biases):
        # squeeze batch dim
        dim = len(weights[0].shape)
        if (weights[0].shape[0] == 1) and (dim > 2):
            weights = tuple([w.squeeze(0) for w in weights])
            biases = tuple([b.squeeze(0) for b in biases])
            dim -= 1

        # add feature dim
        if dim < 3:
            weights = tuple([w.unsqueeze(-1) for w in weights])
            biases = tuple([b.unsqueeze(-1) for b in biases])
        return weights, biases

    def __getitem__(self, item):
        weights, biases, label = self._load_sd_as_weights_biases_label(item)
        if self.do_augmentation:
            weights, biases, label = self._augment(weights, biases, label)
            if self.mixup_augmentation is not None:
                idx = random.randint(0, len(self) - 1)
                weights_1, biases_1, label_1, weight_names, bias_names = self._load_sd_as_weights_biases_label(
                    idx, return_names=True
                )
                mixup_wb = self.mixup_augmentation(
                        weights=weights,
                        biases=biases,
                        weights_rand=weights_1,
                        biases_rand=biases_1,
                        weight_names=weight_names,
                        bias_names=bias_names
                )

                weights = mixup_wb.weights
                biases = mixup_wb.biases
                alpha = mixup_wb.alpha
                label = alpha * F.one_hot(torch.tensor(label), self.num_classes) + (
                    1 - alpha
                ) * F.one_hot(torch.tensor(label_1), self.num_classes)

        weights, biases = self._align_dims(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        return Batch(weights=weights, biases=biases, label=label)
