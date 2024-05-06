import copy
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.inr_models import INR


class GraphProbeFeatures(nn.Module):
    def __init__(self, d_in, num_inputs, inr_model, input_init=None, proj_dim=None):
        super().__init__()
        print(inr_model)
        inr = inr_model
        fmodel, params = make_functional(inr)

        vparams, vshapes = params_to_tensor(params)
        self.sirens = torch.vmap(wrap_func(fmodel, vshapes))

        inputs = input_init if input_init is not None else 2 * torch.rand(1, num_inputs, d_in) - 1
        self.inputs = nn.Parameter(inputs, requires_grad=input_init is None)

        # NOTE hard coded maps
        self.reshape_w = Rearrange("b i j 1 -> b (j i)")
        self.reshape_b = Rearrange("b j 1 -> b j")

        self.proj_dim = proj_dim
        if proj_dim is not None:
            self.proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_inputs, proj_dim),
                    nn.LayerNorm(proj_dim),
                ) for _ in range(len(inr.layers) + 1)])

    def forward(self, weights, biases):
        weights = [self.reshape_w(w) for w in weights]
        biases = [self.reshape_b(b) for b in biases]
        params_flat = torch.cat([w_or_b for p in zip(weights, biases) for w_or_b in p], dim=-1)

        out = self.sirens(params_flat, self.inputs.expand(params_flat.shape[0], -1, -1))
        if self.proj_dim is not None:
            out = [proj(out[i].permute(0, 2, 1)) for i, proj in enumerate(self.proj)]
            out = torch.cat(out, dim=1)
            return out
        else:
            out = torch.cat(out, dim=-1)
            return out.permute(0, 2, 1)


class INRPerLayer(INR):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = [x]
        for layer in self.seq:
            nodes.append(layer(nodes[-1]))
        nodes[-1] = nodes[-1] + 0.5
        return nodes


def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values


def params_to_tensor(params):
    return torch.cat([p.flatten() for p in params]), [p.shape for p in params]


def tensor_to_params(tensor, shapes):
    params = []
    start = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        param = tensor[start: start + size].reshape(shape)
        params.append(param)
        start += size
    return tuple(params)


def wrap_func(func, shapes):
    def wrapped_func(params, *args, **kwargs):
        params = tensor_to_params(params, shapes)
        return func(params, *args, **kwargs)

    return wrapped_func
