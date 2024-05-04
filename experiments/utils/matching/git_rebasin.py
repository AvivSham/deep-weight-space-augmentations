from collections import defaultdict

import copy
from typing import NamedTuple
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import numpy as np
import logging


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int, skip=1) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm(
        {
            "seq.0.weight": ("P_0", None),
            **{
                f"seq.{i*skip}.weight": (f"P_{i}", f"P_{i - 1}")
                for i in range(1, num_hidden_layers)
            },
            **{f"seq.{i*skip}.bias": (f"P_{i}",) for i in range(num_hidden_layers)},
            f"seq.{num_hidden_layers*skip}.weight": (
                None,
                f"P_{num_hidden_layers - 1}",
            ),
            f"seq.{num_hidden_layers*skip}.bias": (None,),
        }
    )


"""
def cnn_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else  {f"{name}.weight": (p_out, p_in)}
  return permutation_spec_from_axes_to_perm({
     **conv("conv1", None, "P_bg0"),
     **conv("conv2", "P_bg0", "P_bg1"),
     **dense("fc1", "P_bg1", "P_bg2"),
     **dense("fc2", "P_bg2", None, False),
  })
"""


def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock("layer1.2", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock("layer2.2", "P_bg2"),
            # **easyblock("layer2.3", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock("layer3.2", "P_bg3"),
            # **easyblock("layer3.3", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("linear", "P_bg3", None),
        }
    )


# should be easy to generalize it to any depth
def resnet50_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock("layer1.2", "P_bg1"),
            **easyblock("layer1.3", "P_bg1"),
            **easyblock("layer1.4", "P_bg1"),
            **easyblock("layer1.5", "P_bg1"),
            **easyblock("layer1.6", "P_bg1"),
            **easyblock("layer1.7", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock("layer2.2", "P_bg2"),
            **easyblock("layer2.3", "P_bg2"),
            **easyblock("layer2.4", "P_bg2"),
            **easyblock("layer2.5", "P_bg2"),
            **easyblock("layer2.6", "P_bg2"),
            **easyblock("layer2.7", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock("layer3.2", "P_bg3"),
            **easyblock("layer3.3", "P_bg3"),
            **easyblock("layer3.4", "P_bg3"),
            **easyblock("layer3.5", "P_bg3"),
            **easyblock("layer3.6", "P_bg3"),
            **easyblock("layer3.7", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("linear", "P_bg3", None),
        }
    )


def vgg16_permutation_spec() -> PermutationSpec:
    layers_with_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    layers_with_conv_b4 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    layers_with_bn = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
    dense = lambda name, p_in, p_out, bias=True: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }
    return permutation_spec_from_axes_to_perm(
        {
            # first features
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),
            "features.1.running_mean": ("P_Conv_0", None),
            "features.1.running_var": ("P_Conv_0", None),
            "features.1.num_batches_tracked": (),
            **{
                f"features.{layers_with_conv[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    f"P_Conv_{layers_with_conv_b4[i]}",
                    None,
                    None,
                )
                for i in range(len(layers_with_conv))
            },
            **{f"features.{i}.bias": (f"P_Conv_{i}",) for i in layers_with_conv + [0]},
            # bn
            **{
                f"features.{layers_with_bn[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.bias": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.running_mean": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.running_var": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.num_batches_tracked": ()
                for i in range(len(layers_with_bn))
            },
            **dense("classifier", "P_Conv_40", "P_Dense_0", False),
        }
    )


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec, params_a, params_b, error=1e-12, max_iter=100, init_perm=None
):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    device = list(params_a.values())[0].device
    perm = (
        {p: torch.arange(n).to(device) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum("ij,ij->i", A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum("ij,ij->i", A, torch.eye(n)[ci, :]).sum()
            # print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + error

            perm[p] = torch.Tensor(ci).to(device)

        if not progress:
            break

    return perm, iteration


def merge_two(model_a, models, num_hidden_layers: int, error=1e-12, skip=1):
    """
    Permute all models in models according to model_a
    :param model_a:
    :param models:
    :param num_hidden_layers:
    :param error:
    :return:
    """

    ps = mlp_permutation_spec(num_hidden_layers=num_hidden_layers, skip=skip)
    # logging.info("start alignment of two models")

    for i, model in enumerate(models):
        perm, num_iters = weight_matching(ps, model_a, model, error=error)

        # duplicate permutation to match weights and biases of each layer
        weight_and_biases_permute = {}
        for p_name, p_permutation in perm.items():
            weight_and_biases_permute[p_name + "_w"] = p_permutation.long()
            weight_and_biases_permute[p_name + "_b"] = p_permutation.long()

        # apply permutation
        prev_layer_perm = None
        layer_index = 0
        for (layer_name, layer_params), layer_perm in zip(
            model.items(), list(weight_and_biases_permute.values()) + [None]
        ):
            layer_index += 1

            # permute columns of weights according to last layer permutations
            if prev_layer_perm is not None and "weight" in layer_name:
                new_params = torch.clone(layer_params)[:, prev_layer_perm, ...]
                layer_params.data.copy_(new_params.data)

            # at the last layer no permutation
            if layer_index > len(model.keys()) - 2:
                break

            # at this layer permute rows
            permuted_params = torch.clone(layer_params)[layer_perm, ...]
            layer_params.data.copy_(permuted_params.data)

            # save permutation of weights to next layer weights
            if "weight" in layer_name:
                prev_layer_perm = torch.clone(layer_perm)

    return perm


def merge_many(models: list, num_hidden_layers: int, error=1e-12, skip=1):
    num_models = len(models)
    ps = mlp_permutation_spec(num_hidden_layers=num_hidden_layers, skip=skip)

    num_iterations = np.asarray([np.inf for k in range(num_models)])
    iters = 0
    logging.info("start alignment")
    # stop when it takes exactly one iteration to converge for all models
    while np.any(num_iterations > 1):
        for i, model in enumerate(models):
            avg_model = copy.deepcopy(model)
            # zero parameters of the average model
            for p in avg_model.values():
                p.data = torch.nn.Parameter(torch.zeros_like(p.data))

            # take the mean not including the current model
            for j, model_b in enumerate(models):
                for layer_name, layer_params in model_b.items():
                    if j != i:
                        avg_model[layer_name].data += layer_params.data / (
                            num_models - 1
                        )

            # PERMUTATIONCOORDINATEDESCENT algorithm
            perm, num_iters = weight_matching(ps, avg_model, model, error=error)
            num_iterations[i] = num_iters

            # duplicate permutation to match weights and biases of each layer
            weight_and_biases_permute = {}
            for p_name, p_permutation in perm.items():
                weight_and_biases_permute[p_name + "_w"] = p_permutation.long()
                weight_and_biases_permute[p_name + "_b"] = p_permutation.long()

            # apply permutation
            prev_layer_perm = None
            layer_index = 0
            # todo: add + [None] or some other way to include the last layer (like copying P_1_b -> P_2_w)
            for (layer_name, layer_params), layer_perm in zip(
                model.items(), list(weight_and_biases_permute.values()) + [None]
            ):
                layer_index += 1

                # permute columns of weights according to last layer permutations
                if prev_layer_perm is not None and "weight" in layer_name:
                    new_params = torch.clone(layer_params)[:, prev_layer_perm, ...]
                    layer_params.data.copy_(new_params.data)

                # at the last layer no permutation
                if layer_index > len(model.keys()) - 2:
                    break

                # at this layer permute rows
                permuted_params = torch.clone(layer_params)[layer_perm, ...]
                layer_params.data.copy_(permuted_params.data)

                # save permutation of weights to next layer weights
                if "weight" in layer_name:
                    prev_layer_perm = torch.clone(layer_perm)

        iters += 1
        logging.info(f"iteration: {iters}, Num. of iters to converge: {num_iterations}")

    # return average model as well
    avg_model = copy.deepcopy(models[0])
    # zero parameters of the average model
    for p in avg_model.values():
        p.data = torch.nn.Parameter(torch.zeros_like(p.data))

    # take the mean not including the current model
    for j, model_b in enumerate(models):
        for layer_name, layer_params in model_b.items():
            avg_model[layer_name].data += layer_params.data / num_models

    return perm, avg_model


def test_weight_matching():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=2)
    print(ps.axes_to_perm)
    rng = torch.Generator()
    rng.manual_seed(13)
    num_hidden = 3
    shapes = {
        "seq.0.weight": (num_hidden * 2, num_hidden),
        "seq.0.bias": (num_hidden * 2,),
        "seq.1.weight": (num_hidden * 3, num_hidden * 2),
        "seq.1.bias": (num_hidden * 3,),
        "seq.2.weight": (num_hidden, num_hidden * 3),
        "seq.2.bias": (num_hidden,),
    }

    params_a = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    params_b = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    # params_b = params_a
    params_b_init = copy.deepcopy(params_b)
    # print(params_b["seq.0.bias"])
    # perm, _ = weight_matching(ps, params_a, params_b)
    # print(perm)
    merge_two(params_a, [params_b], num_hidden_layers=2)
    # print(params_b["seq.0.bias"])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(num_hidden, num_hidden * 2),
                nn.Linear(num_hidden * 2, num_hidden * 3),
                nn.Linear(num_hidden * 3, num_hidden),
            )

        def forward(self, x):
            return self.seq(x)

    x = torch.tensor([[1, 1, 1.0]])
    model = Model()
    model.load_state_dict(params_b_init)
    print(model(x))
    model.load_state_dict(params_b)
    print(model(x))


if __name__ == "__main__":
    test_weight_matching()
