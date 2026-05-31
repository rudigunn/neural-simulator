"""Optional PyTorch Geometric Graph Transformer surrogate."""

from __future__ import annotations

GRAPH_FEATURE_DIM = 12


def ml_dependencies_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


def require_ml_dependencies() -> None:
    if not ml_dependencies_available():
        raise ImportError("Install ML dependencies with `uv sync --group ml`.")


def build_graph_transformer_surrogate(
    *,
    node_feature_dim: int,
    edge_feature_dim: int,
    output_dim: int,
    hidden_dim: int = 64,
    layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    node_encoder_type: str = "schema_attention",
):
    """Build a Graph Transformer surrogate: graph -> KPIs.

    Uses attention-weighted pooling instead of global mean pool.
    """

    try:
        import torch
        from torch import nn
        from torch_geometric.nn import TransformerConv
    except ImportError as exc:
        raise ImportError("Install ML dependencies with `uv sync --group ml`.") from exc

    class LinearNodeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(node_feature_dim, hidden_dim)

        def forward(self, x):
            return self.input(x)

    class SchemaAwareNodeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.field_embedding = nn.Embedding(node_feature_dim, hidden_dim)
            self.value_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.attention = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            field_ids = torch.arange(x.shape[-1], device=x.device)
            field_embeddings = self.field_embedding(field_ids).unsqueeze(0)
            value_embeddings = self.value_encoder(x.unsqueeze(-1))
            cell_embeddings = value_embeddings + field_embeddings
            attention_logits = self.attention(cell_embeddings).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=-1)
            pooled = torch.sum(cell_embeddings * attention_weights.unsqueeze(-1), dim=1)
            return self.output_norm(pooled)

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

    class GraphEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if node_encoder_type == "linear":
                self.node_encoder = LinearNodeEncoder()
            elif node_encoder_type == "schema_attention":
                self.node_encoder = SchemaAwareNodeEncoder()
            else:
                raise ValueError(
                    "node_encoder_type must be one of: linear, schema_attention"
                )
            self.layers = nn.ModuleList(
                [
                    TransformerConv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        edge_dim=edge_feature_dim,
                        beta=True,
                        dropout=dropout,
                    )
                    for _ in range(layers)
                ]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
            self.dropout = nn.Dropout(dropout)
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
            x = self.node_encoder(data.x)
            for conv, norm in zip(self.layers, self.norms):
                residual = x
                x = conv(x, data.edge_index, data.edge_attr)
                x = norm(x + residual)
                x = torch.relu(x)
                x = self.dropout(x)
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

    class GraphTransformerSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = GraphEncoder()
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

    return GraphTransformerSurrogate()


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
