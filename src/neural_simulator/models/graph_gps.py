"""GraphGPS surrogate with Random Walk Structural Encoding (RWSE)."""

from __future__ import annotations

from neural_simulator.models.graph_transformer import GRAPH_FEATURE_DIM


def build_graph_gps_surrogate(
    *,
    node_feature_dim: int,
    edge_feature_dim: int,
    output_dim: int,
    rwse_dim: int = 20,
    hidden_dim: int = 64,
    gps_layers: int = 4,
    heads: int = 4,
    dropout: float = 0.1,
    attn_type: str = "multihead",
    attn_kwargs: dict | None = None,
):
    """Build a GraphGPS surrogate: graph -> KPIs.

    Combines GINEConv (local MPNN with edge features) + a global attention
    block via PyG's GPSConv.  RWSE positional encodings are concatenated to
    raw node features before the input projection.

    ``attn_type`` selects the global attention mechanism:
    - ``"multihead"``: dense softmax attention, O(n^2) per graph (accurate but
      the throughput bottleneck on large graphs).
    - ``"performer"``: linear-attention (FAVOR+) approximation, O(n) per graph.
      This removes the quadratic cost while preserving global message passing,
      which is what makes the surrogate faster than the simulator at scale.
    """

    try:
        import torch
        from torch import nn
        from torch_geometric.nn import GPSConv, GINEConv
    except ImportError as exc:
        raise ImportError("Install ML dependencies with `uv sync --group ml`.") from exc

    class RWSENodeEncoder(nn.Module):
        """Project raw node features + RWSE to hidden_dim."""

        def __init__(self) -> None:
            super().__init__()
            # RWSE features are concatenated to node features
            self.input_proj = nn.Linear(node_feature_dim + rwse_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x, rwse):
            combined = torch.cat([x, rwse], dim=-1)
            return self.norm(self.input_proj(combined))

    class EdgeEncoder(nn.Module):
        """Project edge features to hidden_dim for GINEConv."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(edge_feature_dim, hidden_dim)

        def forward(self, edge_attr):
            return self.proj(edge_attr)

    def _make_gine_conv():
        """Create a GINEConv layer for use inside GPSConv."""
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        return GINEConv(nn=mlp, edge_dim=hidden_dim)

    class AttentionPooling(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, batch):
            gate_logits = self.gate(x).squeeze(-1)
            gate_logits = gate_logits - _scatter_max(gate_logits, batch)
            gate_weights = torch.exp(gate_logits)
            gate_sums = torch.zeros(
                batch.max().item() + 1, device=x.device, dtype=x.dtype
            )
            gate_sums.scatter_add_(0, batch, gate_weights)
            gate_weights = gate_weights / gate_sums[batch].clamp_min(1e-8)
            weighted = x * gate_weights.unsqueeze(-1)
            out = torch.zeros(
                batch.max().item() + 1,
                hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
            return out

    class GraphGPSEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_encoder = RWSENodeEncoder()
            self.edge_encoder = EdgeEncoder()
            self.gps_layers = nn.ModuleList(
                [
                    GPSConv(
                        channels=hidden_dim,
                        conv=_make_gine_conv(),
                        heads=heads,
                        dropout=dropout,
                        act="gelu",
                        norm="layer_norm",
                        attn_type=attn_type,
                        attn_kwargs=attn_kwargs or {},
                    )
                    for _ in range(gps_layers)
                ]
            )
            self.pooling = AttentionPooling()
            self.graph_feature_encoder = nn.Sequential(
                nn.Linear(GRAPH_FEATURE_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def encode_nodes(self, data):
            rwse = getattr(data, "random_walk_pe", None)
            if rwse is None:
                rwse = torch.zeros(
                    data.x.shape[0], rwse_dim, device=data.x.device, dtype=data.x.dtype
                )
            x = self.node_encoder(data.x, rwse)
            edge_attr = self.edge_encoder(data.edge_attr)
            for gps_layer in self.gps_layers:
                x = gps_layer(x, data.edge_index, batch=data.batch, edge_attr=edge_attr)
            return x

        def graph_readout(self, data, x):
            batch = data.batch
            if batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            pooled = self.pooling(x, batch)
            machine_mean, machine_max = _machine_pool_tensors(data, batch, x)
            graph_features = _graph_feature_tensor(data, batch, x)
            graph_embedding = self.graph_feature_encoder(graph_features)
            return self.readout(
                torch.cat([pooled, machine_mean, machine_max, graph_embedding], dim=-1)
            )

        def forward(self, data):
            return self.graph_readout(data, self.encode_nodes(data))

    class GraphGPSSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = GraphGPSEncoder()
            self.kpi_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.machine_utilization_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, data):
            node_embeddings = self.encoder.encode_nodes(data)
            embedding = self.encoder.graph_readout(data, node_embeddings)
            return self.kpi_head(embedding)

        def forward_with_aux(self, data):
            node_embeddings = self.encoder.encode_nodes(data)
            embedding = self.encoder.graph_readout(data, node_embeddings)
            return (
                self.kpi_head(embedding),
                self.machine_utilization_head(node_embeddings).squeeze(-1),
            )

    return GraphGPSSurrogate()


def _scatter_max(values, batch):
    """Per-graph max for numerical stability in softmax."""
    import torch

    num_graphs = batch.max().item() + 1
    out = torch.full((num_graphs,), float("-inf"), device=values.device, dtype=values.dtype)
    out.scatter_reduce_(0, batch, values, reduce="amax")
    return out[batch]


def _graph_feature_tensor(data, batch, x):
    """Return graph-level count features, preserving graph size for absolute KPIs."""
    import torch

    graph_features = getattr(data, "graph_features", None)
    if graph_features is not None:
        if graph_features.dim() == 1:
            graph_features = graph_features.view(-1, GRAPH_FEATURE_DIM)
        return graph_features.to(device=x.device, dtype=x.dtype)

    num_graphs = batch.max().item() + 1
    node_counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    node_counts.scatter_add_(0, batch.unsqueeze(-1), ones)
    zeros = torch.zeros(
        num_graphs,
        GRAPH_FEATURE_DIM - 1,
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([node_counts, zeros], dim=1)


def _machine_pool_tensors(data, batch, x):
    """Mean and max pools over machine nodes only."""
    import torch

    num_graphs = batch.max().item() + 1
    machine_mask = getattr(data, "machine_mask", None)
    if machine_mask is None:
        machine_mask = x.new_zeros(x.shape[0], dtype=torch.bool)
    if not machine_mask.any():
        zeros = x.new_zeros((num_graphs, x.shape[-1]))
        return zeros, zeros

    machine_x = x[machine_mask]
    machine_batch = batch[machine_mask]
    machine_sum = x.new_zeros((num_graphs, x.shape[-1]))
    machine_sum.scatter_add_(
        0,
        machine_batch.unsqueeze(-1).expand_as(machine_x),
        machine_x,
    )
    machine_count = x.new_zeros((num_graphs, 1))
    machine_count.scatter_add_(
        0,
        machine_batch.unsqueeze(-1),
        torch.ones((machine_x.shape[0], 1), device=x.device, dtype=x.dtype),
    )
    machine_mean = machine_sum / machine_count.clamp_min(1.0)

    machine_max = torch.full_like(machine_sum, float("-inf"))
    machine_max.scatter_reduce_(
        0,
        machine_batch.unsqueeze(-1).expand_as(machine_x),
        machine_x,
        reduce="amax",
    )
    machine_max = torch.where(torch.isfinite(machine_max), machine_max, machine_sum)
    return machine_mean, machine_max
