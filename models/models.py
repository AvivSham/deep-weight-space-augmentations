from typing import Tuple

import torch

from torch import nn

from models.layers.layers import (
    CannibalLayer,
    InvariantLayer,
    ReLU,
    DownSampleCannibalLayer,
    Dropout,
    BN,
)


class WSLEncoder(nn.Module):
    def __init__(
        self,
        n_encode_layers,
        input_hidd_dim,
        n_heads,
        hid_ff,
        dropout,
        input_dim_reduced,
        input_in_dim,
    ):
        super().__init__()
        encode_layer = nn.TransformerEncoderLayer(
            d_model=input_hidd_dim,
            nhead=n_heads,
            dim_feedforward=hid_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.lin = nn.Linear(input_hidd_dim, input_hidd_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encode_layer,
            num_layers=n_encode_layers,
            norm=nn.LayerNorm(input_hidd_dim),
        )
        self.input_hidd_dim = input_hidd_dim

        self.input_dim_reduced = input_dim_reduced
        if input_dim_reduced is not None:
            self.input_layer = nn.Linear(input_in_dim, input_dim_reduced)

        self.register_buffer(
            "s_token", torch.randn(1, 1, input_hidd_dim, requires_grad=True)
        )

    def forward(self, x):
        weight, bias = x
        h = bias[0].shape[1]
        weight_ = weight[1:]  # remove first
        bias_ = bias[:-1]  # remove last

        if self.input_dim_reduced is not None:
            w_0 = self.input_layer(weight[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            w_0 = weight[0]

        # each weight is of size (bs, h, h, 1)
        stacked_weight = torch.cat(
            [w.squeeze(-1).permute(0, 2, 1) for w in weight_], dim=1
        )  # (bs, h*(L-2), h)
        # each weight is of size (bs, h, 1)
        stacked_bias = torch.cat(
            [w.permute(0, 2, 1) for w in bias_], dim=1
        )  # (bs, (L-2), h)

        # adding first weight matrix - we transpose as mentioned in INR2Vec
        stacked_weight = torch.cat(
            [stacked_weight, w_0.squeeze(-1)], dim=1
        )  # (bs, d0 + h*(L-2), h)

        # adding last bias - we repeat (broadcast) as mentioned in INR2Vec
        # last bias is (bs, dM, 1)
        stacked_weight = torch.cat([stacked_weight, bias[-1].repeat(1, 1, h)], dim=1)

        model_input = torch.cat(
            [stacked_weight, stacked_bias], dim=1
        )  # (bs, h*(L-2)+h, h)
        # model_input = model_input.permute(0, 2, 1)
        x = self.lin(model_input)
        x = torch.cat([self.s_token.repeat(x.shape[0], 1, 1), x], dim=1)
        return self.transformer_encoder(x)[:, 0, :]  # slicing CLS token


class WSLForClassification(WSLEncoder):
    def __init__(
        self,
        n_encode_layers,
        input_hidd_dim,
        n_heads,
        hid_ff,
        dropout,
        n_classes,
        input_dim_reduced=None,
        input_in_dim=None,
    ):
        super().__init__(
            n_encode_layers,
            input_hidd_dim,
            n_heads,
            hid_ff,
            dropout,
            input_dim_reduced,
            input_in_dim,
        )
        self.l = nn.Linear(input_hidd_dim, n_classes)

    def forward(self, x):
        return self.l(super().forward(x).squeeze(1))


class WSLForModel(WSLEncoder):
    def __init__(
        self,
        n_encode_layers,
        input_hidd_dim,
        n_heads,
        hid_ff,
        dropout,
        n_classes,
        encoder_out_dim=None,
        input_dim_reduced=None,
        input_in_dim=None,
    ):
        super().__init__(
            n_encode_layers,
            input_hidd_dim,
            n_heads,
            hid_ff,
            dropout,
            input_dim_reduced,
            input_in_dim,
        )

        self.proj_l = None

        if encoder_out_dim is not None:
            self.proj_l = nn.Linear(input_hidd_dim, encoder_out_dim)
            self.l = nn.Linear(encoder_out_dim, n_classes)
        else:
            self.l = nn.Linear(input_hidd_dim, n_classes)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weight, bias = x
        bs = weight[0].shape[0]
        weight_shape, bias_shape = [w[0, :].shape for w in weight], [
            b[0, :].shape for b in bias
        ]
        e_out = super().forward(x).squeeze(1)
        if self.proj_l is not None:
            e_out = self.proj_l(e_out)
        weights_and_biases = self.l(e_out)

        n_weights = sum([w.numel() for w in weight_shape])
        weights = weights_and_biases[:, :n_weights]
        biases = weights_and_biases[:, n_weights:]
        weight, bias = [], []
        w_index = 0
        for s in weight_shape:
            weight.append(weights[:, w_index : w_index + s.numel()].reshape(bs, *s))
            w_index += s.numel()
        w_index = 0
        for s in bias_shape:
            bias.append(biases[:, w_index : w_index + s.numel()].reshape(bs, *s))
            w_index += s.numel()
        return tuple(weight), tuple(bias)


class FCBaselineForClassification(nn.Module):
    def __init__(self, in_dim=2208, hidden_dim=256, n_hidden=2, n_classes=10, bn=False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden):
            if not bn:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    ]
                )

        layers.append(nn.Linear(hidden_dim, n_classes))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weight, bias = x
        all_weights = weight + bias
        weight = torch.cat([w.flatten(start_dim=1) for w in all_weights], dim=-1)
        return self.seq(weight)


class CannibalModel(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        input_features,
        hidden_dim,
        n_hidden=2,
        output_features=None,
        reduction="max",
        bias=True,
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        input_dim_downsample=None,
        dropout_rate=0.0,
        add_skip=False,
        add_layer_skip=False,
        init_scale=1e-4,
        init_off_diag_scale_penalty=1.0,
        bn=False,
        diagonal=False,
    ):
        super().__init__()
        assert (
            len(weight_shapes) > 2
        ), "the current implementation only support input networks with M>2 layers."

        self.input_features = input_features
        self.input_dim_downsample = input_dim_downsample
        if output_features is None:
            output_features = hidden_dim

        self.add_skip = add_skip
        if self.add_skip:
            self.skip = nn.Linear(input_features, output_features, bias=bias)
            with torch.no_grad():
                torch.nn.init.constant_(
                    self.skip.weight, 1.0 / self.skip.weight.numel()
                )
                torch.nn.init.constant_(self.skip.bias, 0.0)

        if input_dim_downsample is None:
            layers = [
                CannibalLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [
                        ReLU(),
                        Dropout(dropout_rate),
                        CannibalLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim
                            if i != (n_hidden - 1)
                            else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        else:
            layers = [
                DownSampleCannibalLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    downsample_dim=input_dim_downsample,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [
                        ReLU(),
                        Dropout(dropout_rate),
                        DownSampleCannibalLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim
                            if i != (n_hidden - 1)
                            else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            downsample_dim=input_dim_downsample,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        out = self.layers(x)
        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(
                self.skip(b) for b in x[1]
            )
            weight_out = tuple(ws + w for w, ws in zip(out[0], skip_out[0]))
            bias_out = tuple(bs + b for b, bs in zip(out[1], skip_out[1]))
            out = weight_out, bias_out
        return out


class CannibalModelForClassification(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        input_features,
        hidden_dim,
        n_hidden=2,
        n_classes=10,
        reduction="max",
        bias=True,
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        n_out_fc=1,
        dropout_rate=0.0,
        input_dim_downsample=None,
        init_scale=1.0,
        init_off_diag_scale_penalty=1.0,
        bn=False,
        add_skip=False,
        add_layer_skip=False,
        equiv_out_features=None,
        diagonal=False,
    ):
        super().__init__()
        self.layers = CannibalModel(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=input_features,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            reduction=reduction,
            bias=bias,
            output_features=equiv_out_features,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
            dropout_rate=dropout_rate,
            input_dim_downsample=input_dim_downsample,
            init_scale=init_scale,
            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
            bn=bn,
            add_skip=add_skip,
            add_layer_skip=add_layer_skip,
            diagonal=diagonal,
        )
        self.dropout = Dropout(dropout_rate)
        self.relu = ReLU()
        self.clf = InvariantLayer(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            in_features=hidden_dim
            if equiv_out_features is None
            else equiv_out_features,
            out_features=n_classes,
            reduction=reduction,
            n_fc_layers=n_out_fc,
        )

    def forward(
        self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]], return_equiv=False
    ):
        x = self.layers(x)
        out = self.clf(self.dropout(self.relu(x)))
        if return_equiv:
            return out, x
        else:
            return out


class CannibalModelFeatureMixup(CannibalModelForClassification):
    def forward(
            self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]], return_equiv=False
    ):
        if return_equiv:
            x = self.layers(x)
            return x
        else:
            out = self.clf(self.dropout(self.relu(x)))
            return out
