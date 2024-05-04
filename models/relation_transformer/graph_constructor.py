import math
import torch
import torch.nn as nn

from models.relation_transformer.probe_features import GraphProbeFeatures


def batch_to_graphs(weights, biases, weights_mean=None, weights_std=None,
                    biases_mean=None, biases_std=None):
    # assume graph has (2 + 32 + 32 + 1) [features] + (32 + 32 + 1) [bias] = 132 nodes
    device = weights[0].device
    bsz = weights[0].shape[0]
    num_nodes = weights[0].shape[1] + sum(w.shape[2] for w in weights)

    node_features = torch.zeros(bsz, num_nodes, biases[0].shape[-1], device=device)
    edge_features = torch.zeros(bsz, num_nodes, num_nodes, weights[0].shape[-1], device=device)

    row_offset = 0
    col_offset = weights[0].shape[1]  # no edge to input nodes
    for i, w in enumerate(weights):
        _, num_in, num_out, _ = w.shape
        w_mean = weights_mean[i] if weights_mean is not None else 0
        w_std = weights_std[i] if weights_std is not None else 1
        edge_features[
        :, row_offset: row_offset + num_in, col_offset: col_offset + num_out
        ] = (w - w_mean) / w_std
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[1]  # no bias in input nodes
    for i, b in enumerate(biases):
        _, num_out, _ = b.shape
        b_mean = biases_mean[i] if biases_mean is not None else 0
        b_std = biases_std[i] if biases_std is not None else 1
        node_features[:, row_offset: row_offset + num_out] = (b - b_mean) / b_std
        row_offset += num_out

    return node_features, edge_features


class GaussianFourierFeatureTransform(nn.Module):
    """
    Given an input of size [batches, num_input_channels, ...],
     returns a tensor of size [batches, mapping_size*2, ...].
    """

    def __init__(self, in_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = in_channels
        self._mapping_size = mapping_size
        self.out_channels = mapping_size * 2
        self.register_buffer("_B", torch.randn((in_channels, mapping_size)) * scale)

    def forward(self, x):
        assert len(x.shape) >= 3

        x = (x @ self._B)
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class GraphConstructor(nn.Module):
    def __init__(self,
                 d_in,
                 d_node,
                 d_edge,
                 layer_layout,
                 rev_edge_features=False,
                 zero_out_bias=False,
                 zero_out_weights=False,
                 inp_factor=1,
                 input_layers=1,
                 sin_emb=False,
                 use_pos_embed=True,
                 num_probe_features=0,
                 inr_model=None,
                 ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.zero_out_bias = zero_out_bias
        self.zero_out_weights = zero_out_weights
        self.use_pos_embed = use_pos_embed

        self.pos_embed_layout = (
                [1] * layer_layout[0] + layer_layout[1:-1] + [1] * layer_layout[-1]
        )
        self.pos_embed = nn.Parameter(torch.randn(len(self.pos_embed_layout), d_node))

        proj_weight = []
        proj_bias = []
        if sin_emb:
            proj_weight.append(GaussianFourierFeatureTransform(d_in + (2 if rev_edge_features else 0), 128, inp_factor))
            # proj_weight.append(SinusoidalPosEmb(4*d_hid, fact=inp_factor))
            proj_weight.append(nn.Linear(256, d_edge))
            proj_bias.append(GaussianFourierFeatureTransform(d_in, 128, inp_factor))
            # proj_bias.append(SinusoidalPosEmb(4*d_hid, fact=inp_factor))
            proj_bias.append(nn.Linear(256, d_node))
        else:
            proj_weight.append(nn.Linear(d_in + (2 if rev_edge_features else 0), d_edge))
            proj_bias.append(nn.Linear(d_in, d_node))
            # proj_weight.append(nn.LayerNorm(d_hid))
            # proj_bias.append(nn.LayerNorm(d_hid))

        for i in range(input_layers - 1):
            proj_weight.append(nn.SiLU())
            proj_weight.append(nn.Linear(d_edge, d_edge))
            proj_bias.append(nn.SiLU())
            proj_bias.append(nn.Linear(d_node, d_node))

        self.proj_weight = nn.Sequential(*proj_weight)
        self.proj_bias = nn.Sequential(*proj_bias)

        self.proj_node_in = nn.Linear(d_node, d_node)
        self.proj_edge_in = nn.Linear(d_edge, d_edge)

        if num_probe_features > 0:
            self.gpf = GraphProbeFeatures(
                d_in=layer_layout[0],
                num_inputs=num_probe_features,
                inr_model=inr_model,
                input_init=None,
                proj_dim=d_node,
            )
        else:
            self.gpf = None

    def forward(self, inputs):
        node_features, edge_features = batch_to_graphs(*inputs)
        # mask currently unused
        mask = edge_features.sum(dim=-1, keepdim=True) != 0
        # mask = mask & mask.transpose(-1, -2)
        if self.rev_edge_features:
            # NOTE doesn't work together with other features anymore
            rev_edge_features = edge_features.transpose(-2, -3)
            edge_features = torch.cat(
                [edge_features, rev_edge_features, edge_features + rev_edge_features],
                dim=-1,
            )

        node_features = self.proj_bias(node_features)
        edge_features = self.proj_weight(edge_features)

        if self.zero_out_weights:
            edge_features = torch.zeros_like(edge_features)
        if self.zero_out_bias:
            # only zero out bias, not gpf
            node_features = torch.zeros_like(node_features)

        if self.gpf is not None:
            probe_features = self.gpf(*inputs)
            node_features = node_features + probe_features

        node_features = self.proj_node_in(node_features)
        edge_features = self.proj_edge_in(edge_features)

        if self.use_pos_embed:
            pos_embed = torch.cat(
                [
                    # repeat(self.pos_embed[i], "d -> 1 n d", n=n)
                    self.pos_embed[i].unsqueeze(0).expand(1, n, -1)
                    for i, n in enumerate(self.pos_embed_layout)
                ],
                dim=1,
            )
            node_features = node_features + pos_embed
        return node_features, edge_features, mask
