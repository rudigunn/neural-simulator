"""Multi-scale type-aware pooling with Set2Set readout.

Hypothesis: separately pooling machine / order / operation nodes preserves
structural information that global pooling destroys, improving flow_time
and tardiness predictions on out-of-distribution topologies.
"""

from __future__ import annotations

from neural_simulator.models.graph_transformer import GRAPH_FEATURE_DIM


def build_multiscale_pool_surrogate(
    *,
    node_feature_dim: int,
    edge_feature_dim: int,
    output_dim: int,
    hidden_dim: int = 64,
    layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    set2set_processing_steps: int = 4,
    node_encoder_type: str = "schema_attention",
):
    """Build a multi-scale type-aware pooling surrogate with Set2Set readout.

    Architecture improvements over the baseline GraphTransformerSurrogate:
    1. GATv2Conv layers (dynamic attention) instead of TransformerConv.
    2. Multi-scale type-aware readout: pool machine/order/operation nodes
       SEPARATELY, then concatenate.
    3. Intermediate-layer readout: pool from layers 1, 2, AND 3 (not just
       the final layer).
    4. Set2Set aggregation instead of attention pooling for a permutation-
       invariant readout that captures set statistics.
    """

    try:
        import torch
        from torch import nn
        from torch_geometric.nn import GATv2Conv
        from torch_geometric.nn.aggr import Set2Set
    except ImportError as exc:
        raise ImportError(
            "Install ML dependencies with `uv sync --group ml`."
        ) from exc

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
            pooled = torch.sum(
                cell_embeddings * attention_weights.unsqueeze(-1), dim=1
            )
            return self.output_norm(pooled)

    class LinearNodeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(node_feature_dim, hidden_dim)

        def forward(self, x):
            return self.input(x)

    class TypeAwareSet2SetPool(nn.Module):
        """Pool nodes of a specific type using Set2Set aggregation."""

        def __init__(self) -> None:
            super().__init__()
            self.set2set = Set2Set(hidden_dim, processing_steps=set2set_processing_steps)

        def forward(self, x, batch, mask):
            """Pool only nodes where mask is True.

            Returns a (num_graphs, 2 * hidden_dim) tensor.
            Missing types for a graph get zeros.
            """
            num_graphs = batch.max().item() + 1
            output_dim_local = 2 * hidden_dim

            if not mask.any():
                return x.new_zeros((num_graphs, output_dim_local))

            masked_x = x[mask]
            masked_batch = batch[mask]
            pooled = self.set2set(masked_x, masked_batch)

            # Set2Set may return fewer graphs if some graphs have no nodes
            # of this type. Scatter into full-size output.
            unique_graphs = masked_batch.unique()
            if pooled.shape[0] == num_graphs:
                return pooled

            full_output = x.new_zeros((num_graphs, output_dim_local))
            full_output[unique_graphs] = pooled
            return full_output

    class GlobalSet2SetPool(nn.Module):
        """Global Set2Set over all nodes."""

        def __init__(self) -> None:
            super().__init__()
            self.set2set = Set2Set(hidden_dim, processing_steps=set2set_processing_steps)

        def forward(self, x, batch):
            return self.set2set(x, batch)

    class MultiscaleGraphEncoder(nn.Module):
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

            # GATv2Conv message passing layers
            self.conv_layers = nn.ModuleList(
                [
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        edge_dim=edge_feature_dim,
                        dropout=dropout,
                        add_self_loops=True,
                        share_weights=False,
                    )
                    for _ in range(layers)
                ]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(layers)]
            )
            self.dropout = nn.Dropout(dropout)

            # Global Set2Set pooling (for each intermediate layer)
            self.global_pools = nn.ModuleList(
                [GlobalSet2SetPool() for _ in range(layers)]
            )

            # Type-aware Set2Set pooling (applied to final layer)
            self.machine_pool = TypeAwareSet2SetPool()
            self.order_pool = TypeAwareSet2SetPool()
            self.operation_pool = TypeAwareSet2SetPool()

            # Graph feature encoder
            self.graph_feature_encoder = nn.Sequential(
                nn.Linear(GRAPH_FEATURE_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # Readout dimensions:
            # - Global Set2Set per layer: layers * (2 * hidden_dim)
            # - Type-aware pools: 3 * (2 * hidden_dim)
            # - Graph features: hidden_dim
            readout_input_dim = (
                layers * (2 * hidden_dim)    # intermediate layer global pools
                + 3 * (2 * hidden_dim)       # type-aware pools
                + hidden_dim                 # graph features
            )

            self.readout = nn.Sequential(
                nn.Linear(readout_input_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def encode_nodes(self, data):
            """Run message passing and collect intermediate representations."""
            x = self.node_encoder(data.x)
            intermediate_xs = []
            for conv, norm in zip(self.conv_layers, self.norms):
                residual = x
                x = conv(x, data.edge_index, data.edge_attr)
                x = norm(x + residual)
                x = torch.relu(x)
                x = self.dropout(x)
                intermediate_xs.append(x)
            return x, intermediate_xs

        def graph_readout(self, data, x, intermediate_xs):
            batch = data.batch
            if batch is None:
                batch = torch.zeros(
                    x.shape[0], dtype=torch.long, device=x.device
                )

            # Intermediate-layer global pooling
            layer_pools = []
            for layer_x, pool in zip(intermediate_xs, self.global_pools):
                layer_pools.append(pool(layer_x, batch))

            # Type-aware pooling on final layer
            machine_mask = getattr(data, "machine_mask", None)
            order_mask = getattr(data, "order_mask", None)
            operation_mask = getattr(data, "operation_mask", None)

            if machine_mask is None:
                machine_mask = x.new_zeros(x.shape[0], dtype=torch.bool)
            if order_mask is None:
                order_mask = x.new_zeros(x.shape[0], dtype=torch.bool)
            if operation_mask is None:
                operation_mask = x.new_zeros(x.shape[0], dtype=torch.bool)

            machine_pooled = self.machine_pool(x, batch, machine_mask)
            order_pooled = self.order_pool(x, batch, order_mask)
            operation_pooled = self.operation_pool(x, batch, operation_mask)

            # Graph-level features
            graph_features = _graph_feature_tensor(data, batch, x)
            graph_embedding = self.graph_feature_encoder(graph_features)

            # Concatenate all representations
            all_features = (
                layer_pools
                + [machine_pooled, order_pooled, operation_pooled, graph_embedding]
            )
            combined = torch.cat(all_features, dim=-1)
            return self.readout(combined)

        def forward(self, data):
            x, intermediate_xs = self.encode_nodes(data)
            return self.graph_readout(data, x, intermediate_xs)

    class MultiscalePoolSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = MultiscaleGraphEncoder()
            self.kpi_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.machine_utilization_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, data):
            x, intermediate_xs = self.encoder.encode_nodes(data)
            embedding = self.encoder.graph_readout(data, x, intermediate_xs)
            return self.kpi_head(embedding)

        def forward_with_aux(self, data):
            x, intermediate_xs = self.encoder.encode_nodes(data)
            embedding = self.encoder.graph_readout(data, x, intermediate_xs)
            return (
                self.kpi_head(embedding),
                self.machine_utilization_head(x).squeeze(-1),
            )

    return MultiscalePoolSurrogate()


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
