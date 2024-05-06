import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from models.relation_transformer.graph_constructor import GraphConstructor


class RelationTransformer(nn.Module):
    def __init__(
            self,
            d_in,
            layer_layout,
            inr_model,
            d_node=64,
            d_edge=32,
            d_attn_hid=128,
            d_node_hid=128,
            d_edge_hid=64,
            d_out_hid=128,
            d_out=3,
            n_layers=4,
            n_heads=8,
            num_probe_features=0,
            dropout_node=0.2,
            dropout_edge=0.2,
            node_update_type="rt",
            disable_edge_updates=False,
            use_cls_token=False,
            graph_features="cat_last_layer",
            rev_edge_features=False,
            zero_out_bias=False,
            zero_out_weights=False,
            sin_emb=True,
            input_layers=1,
            use_pos_embed=True,
            use_topomask=False,
            inp_factor=1.0,
            modulate_v=True,
            use_ln=True,
            tfixit_init=False,
    ):
        super().__init__()
        assert use_cls_token == (graph_features == "cls_token")
        self.graph_features = graph_features
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.construct_graph = GraphConstructor(
            d_in=d_in,
            d_node=d_node,
            d_edge=d_edge,
            layer_layout=layer_layout,
            rev_edge_features=rev_edge_features,
            zero_out_bias=zero_out_bias,
            zero_out_weights=zero_out_weights,
            sin_emb=sin_emb,
            use_pos_embed=use_pos_embed,
            input_layers=input_layers,
            inp_factor=inp_factor,
            num_probe_features=num_probe_features,
            inr_model=inr_model,
        )
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(d_node))

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        d_node,
                        d_edge,
                        d_attn_hid,
                        d_node_hid,
                        d_edge_hid,
                        n_heads,
                        dropout_node,
                        dropout_edge,
                        node_update_type=node_update_type,
                        disable_edge_updates=(
                                (disable_edge_updates or (i == n_layers - 1))
                                and (graph_features != "mean_edge")),
                        use_topomask=use_topomask,
                        modulate_v=modulate_v,
                        use_ln=use_ln,
                        tfixit_init=tfixit_init,
                        n_layers=n_layers,
                    )
                )
                for i in range(n_layers)
            ]
        )
        num_graph_features = (
            layer_layout[-1] * d_node if graph_features == "cat_last_layer" else d_node
        )
        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_out_hid),
            nn.ReLU(),
            # nn.Linear(d_out_hid, d_out_hid),
            # nn.ReLU(),
            nn.Linear(d_out_hid, d_out),
        )

    def forward(self, inputs):
        node_features, edge_features, mask = self.construct_graph(inputs)
        if self.use_cls_token:
            node_features = torch.cat(
                [
                    # repeat(self.cls_token, "d -> b 1 d", b=node_features.size(0)),
                    self.cls_token.unsqueeze(0).expand(node_features.size(0), 1, -1),
                    node_features,
                ],
                dim=1,
            )
            edge_features = F.pad(edge_features, (0, 0, 1, 0, 1, 0), value=0)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, mask)

        if self.graph_features == "cls_token":
            graph_features = node_features[:, 0]
        elif self.graph_features == "mean":
            graph_features = node_features.mean(dim=1)
        elif self.graph_features == "max":
            graph_features = node_features.max(dim=1).values
        elif self.graph_features == "last_layer":
            graph_features = node_features[:, -self.nodes_per_layer[-1]:].mean(dim=1)
        elif self.graph_features == "cat_last_layer":
            graph_features = node_features[:, -self.nodes_per_layer[-1]:].flatten(1, 2)
        elif self.graph_features == "mean_edge":
            graph_features = edge_features.mean(dim=(1, 2))
        elif self.graph_features == "max_edge":
            graph_features = edge_features.flatten(1, 2).max(dim=1).values
        elif self.graph_features == "last_layer_edge":
            graph_features = edge_features[:, -self.nodes_per_layer[-1]:, :].mean(
                dim=(1, 2)
            )

        return self.proj_out(graph_features)


class RTLayer(nn.Module):
    def __init__(
            self,
            d_node,
            d_edge,
            d_attn_hid,
            d_node_hid,
            d_edge_hid,
            n_heads,
            dropout_node,
            dropout_edge,
            node_update_type="rt",
            disable_edge_updates=False,
            use_topomask=False,
            modulate_v=True,
            use_ln=True,
            tfixit_init=False,
            n_layers=None,
    ):
        super().__init__()
        self.node_update_type = node_update_type
        self.disable_edge_updates = disable_edge_updates
        self.use_ln = use_ln
        self.n_layers = n_layers

        self.self_attn = torch.jit.script(
            RTAttention(d_node, d_edge, d_attn_hid, n_heads, use_topomask=use_topomask,
                        modulate_v=modulate_v, use_ln=use_ln))
        # self.self_attn = RTAttention(d_hid, d_hid, d_hid, n_heads)
        self.lin0 = Linear(d_node, d_node)
        self.dropout0 = nn.Dropout(dropout_node)
        if use_ln:
            self.node_ln0 = nn.LayerNorm(d_node)
            self.node_ln1 = nn.LayerNorm(d_node)
        else:
            self.node_ln0 = nn.Identity()
            self.node_ln1 = nn.Identity()

        act_fn = nn.GELU

        self.node_mlp = nn.Sequential(
            Linear(d_node, d_node_hid, bias=False),
            act_fn(),
            Linear(d_node_hid, d_node),
            nn.Dropout(dropout_node),
        )

        if not self.disable_edge_updates:
            self.edge_updates = EdgeLayer(d_node=d_node, d_edge=d_edge, d_edge_hid=d_edge_hid,
                                          dropout=dropout_edge, act_fn=act_fn, use_ln=use_ln)
        else:
            self.edge_updates = NoEdgeLayer()

        if tfixit_init:
            self.fixit_init()

    def fixit_init(self):
        temp_state_dict = self.state_dict()
        n_layers = self.n_layers
        for name, param in self.named_parameters():
            if "weight" in name:
                if name.split(".")[0] in ["node_mlp", "edge_mlp0", "edge_mlp1"]:
                    temp_state_dict[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * param
                elif name.split(".")[0] in ["self_attn"]:
                    temp_state_dict[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * (param * (2 ** 0.5))

        self.load_state_dict(temp_state_dict)

    def node_updates(self, node_features, edge_features, mask):
        # attn_out = checkpoint(self.self_attn, node_features, edge_features, mask)
        node_features = self.node_ln0(
            node_features
            + self.dropout0(
                self.lin0(self.self_attn(node_features, edge_features, mask))
            )
        )
        node_features = self.node_ln1(node_features + self.node_mlp(node_features))

        return node_features

    def forward(self, node_features, edge_features, mask):
        node_features = self.node_updates(node_features, edge_features, mask)
        edge_features = self.edge_updates(node_features, edge_features, mask)

        return node_features, edge_features


class EdgeLayer(nn.Module):
    def __init__(self,
                 *,
                 d_node,
                 d_edge,
                 d_edge_hid,
                 dropout,
                 act_fn,
                 use_ln=True,
                 ) -> None:
        super().__init__()
        self.edge_mlp0 = EdgeMLP(
            d_edge=d_edge, d_node=d_node, d_edge_hid=d_edge_hid,
            act_fn=act_fn, dropout=dropout)
        self.edge_mlp1 = nn.Sequential(
            Linear(d_edge, d_edge_hid, bias=False),
            act_fn(),
            Linear(d_edge_hid, d_edge),
            nn.Dropout(dropout),
        )
        if use_ln:
            self.eln0 = nn.LayerNorm(d_edge)
            self.eln1 = nn.LayerNorm(d_edge)
        else:
            self.eln0 = nn.Identity()
            self.eln1 = nn.Identity()

    def forward(self, node_features, edge_features, mask):
        edge_features = self.eln0(edge_features + self.edge_mlp0(node_features, edge_features))
        edge_features = self.eln1(edge_features + self.edge_mlp1(edge_features))
        return edge_features


class NoEdgeLayer(nn.Module):
    def forward(self, node_features, edge_features, mask):
        return edge_features


class EdgeMLP(nn.Module):
    def __init__(self, *, d_node, d_edge, d_edge_hid, act_fn, dropout):
        super().__init__()
        # self.d_hid = d_hid
        self.reverse_edge = Rearrange("b n m d -> b m n d")
        self.lin0_e = Linear(2 * d_edge, d_edge_hid)
        self.lin0_s = Linear(d_node, d_edge_hid)
        self.lin0_t = Linear(d_node, d_edge_hid)
        # self.lin0_er = Linear(d_hid, d_hid, bias=False)
        # self.lin0_ec = Linear(d_hid, d_hid, bias=False)
        self.act = act_fn()
        self.lin1 = Linear(d_edge_hid, d_edge)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_features, edge_features):
        source_nodes = self.lin0_s(node_features).unsqueeze(-2).expand(
            -1, -1, node_features.size(-2), -1
        )
        target_nodes = self.lin0_t(node_features).unsqueeze(-3).expand(
            -1, node_features.size(-2), -1, -1
        )
        # source_edge = self.lin0_ec(edge_features.mean(dim=-3, keepdim=True)).expand(
        #     -1, edge_features.size(-3), -1, -1
        # )
        # target_edge = self.lin0_er(edge_features.mean(dim=-2, keepdim=True)).expand(
        #     -1, -1, edge_features.size(-2), -1
        # )

        # reversed_edge_features = self.reverse_edge(edge_features)
        edge_features = self.lin0_e(torch.cat([edge_features, self.reverse_edge(edge_features)], dim=-1))
        edge_features = edge_features + source_nodes + target_nodes  # + source_edge + target_edge
        edge_features = self.act(edge_features)
        edge_features = self.lin1(edge_features)
        edge_features = self.drop(edge_features)

        return edge_features


class RTAttention(nn.Module):
    def __init__(self, d_node, d_edge, d_hid, n_heads, use_topomask=False, modulate_v=None,
                 use_ln=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hid = d_hid
        self.use_ln = use_ln
        self.modulate_v = modulate_v
        self.scale = 1 / (d_hid ** 0.5)
        self.split_head_node = Rearrange("b n (h d) -> b h n d", h=n_heads)
        self.split_head_edge = Rearrange("b n m (h d) -> b h n m d", h=n_heads)
        self.cat_head_node = Rearrange("... h n d -> ... n (h d)", h=n_heads)

        self.qkv_node = Linear(d_node, 3 * d_hid, bias=False)
        self.edge_factor = 4 if modulate_v else 3
        self.qkv_edge = Linear(d_edge, self.edge_factor * d_hid, bias=False)
        self.proj_out = Linear(d_hid, d_node)
        # if use_ln:
        #     self.ln_q = nn.LayerNorm(d_hid // n_heads)
        #     self.ln_k = nn.LayerNorm(d_hid // n_heads)
        # else:
        #     self.ln_q = nn.Identity()
        #     self.ln_k = nn.Identity()

    def forward(self, node_features, edge_features, mask):
        qkv_node = self.qkv_node(node_features)
        # qkv_node = rearrange(qkv_node, "b n (h d) -> b h n d", h=self.n_heads)
        qkv_node = self.split_head_node(qkv_node)
        q_node, k_node, v_node = torch.chunk(qkv_node, 3, dim=-1)

        qkv_edge = self.qkv_edge(edge_features)
        # qkv_edge = rearrange(qkv_edge, "b n m (h d) -> b h n m d", h=self.n_heads)
        qkv_edge = self.split_head_edge(qkv_edge)
        qkv_edge = torch.chunk(qkv_edge, self.edge_factor, dim=-1)
        # q_edge, k_edge, v_edge, q_edge_b, k_edge_b, v_edge_b = torch.chunk(
        #     qkv_edge, 6, dim=-1
        # )

        q = q_node.unsqueeze(-2) + qkv_edge[0]  # + q_edge_b
        k = k_node.unsqueeze(-3) + qkv_edge[1]  # + k_edge_b
        if self.modulate_v:
            v = v_node.unsqueeze(-3) * qkv_edge[3] + qkv_edge[2]
        else:
            v = v_node.unsqueeze(-3) + qkv_edge[2]

        # q = self.ln_q(q)
        # k = self.ln_k(k)
        dots = self.scale * torch.einsum("b h i j d, b h i j d -> b h i j", q, k)

        attn = F.softmax(dots, dim=-1)
        out = torch.einsum("b h i j, b h i j d -> b h i d", attn, v)
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.cat_head_node(out)
        return self.proj_out(out)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)  # , gain=1 / math.sqrt(2))
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m