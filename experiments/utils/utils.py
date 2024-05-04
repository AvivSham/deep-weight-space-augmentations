import argparse
import json
import logging
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import wandb


def parse_aug(v):
    if v is not None:
        return v.replace(" ", "").split(",")
    else:
        return v


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def reshape_model_output(x, y_shape, x_shape, bs):
    return x.view(bs, y_shape, x_shape, -1).permute(0, 3, 1, 2)


def get_art_dir(args):
    art_dir = Path(args.out_dir)
    art_dir.mkdir(exist_ok=True, parents=True)

    curr = 0
    existing = [
        int(x.as_posix().split("_")[-1]) for x in art_dir.iterdir() if x.is_dir()
    ]
    if len(existing) > 0:
        curr = max(existing) + 1

    out_dir = art_dir / f"version_{curr}"
    out_dir.mkdir()

    return out_dir


def save_experiment(args, results, return_out_dir=False, save_results=True):
    out_dir = get_art_dir(args)

    json.dump(vars(args), open(out_dir / "meta.experiment", "w"))

    # loss curve
    if save_results:
        json.dump(results, open(out_dir / "results.experiment", "w"))

    if return_out_dir:
        return out_dir


def model_save(model, file=None, log_to_wandb=False):
    if file is None:
        file = BytesIO()
    torch.save({"model_state_dict": model.state_dict()}, file)
    if log_to_wandb:
        wandb.save(file.as_posix())

    return file


def model_load(model, file):
    if isinstance(file, BytesIO):
        file.seek(0)

    model.load_state_dict(
        torch.load(file, map_location=lambda storage, location: storage)[
            "model_state_dict"
        ]
    )

    return model


def save_data(tensor, file, log_to_wandb=False):
    torch.save(tensor, file)
    if log_to_wandb:
        wandb.save(file.as_posix())


def load_data(file):
    return torch.load(file, map_location=lambda storage, location: storage)


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
