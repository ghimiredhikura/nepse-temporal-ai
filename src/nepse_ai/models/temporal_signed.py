"""Role-aware temporal broker-security stress model."""

from __future__ import annotations

import torch


class TemporalStressModel(torch.nn.Module):
    """GRU security memory with an optional signed transaction encoder."""

    def __init__(
        self,
        model_type: str,
        tabular_dimension: int,
        broker_count: int,
        edge_dimension: int = 9,
        embedding_dimension: int = 32,
        message_dimension: int = 32,
        hidden_dimension: int = 64,
        dropout: float = 0.10,
        graph_aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if model_type not in {"temporal_tabular", "temporal_signed"}:
            raise ValueError(f"Unsupported model type: {model_type}")
        if graph_aggregation not in {"mean", "value_weighted"}:
            raise ValueError(
                "graph_aggregation must be 'mean' or 'value_weighted'"
            )
        self.model_type = model_type
        self.graph_aggregation = graph_aggregation
        self.message_dimension = message_dimension
        self.hidden_dimension = hidden_dimension
        self.broker_embedding = torch.nn.Embedding(
            broker_count, embedding_dimension
        )
        self.message = torch.nn.Sequential(
            torch.nn.Linear(
                2 * embedding_dimension + edge_dimension, 64
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, message_dimension),
            torch.nn.ReLU(),
        )
        graph_dimension = (
            message_dimension if model_type == "temporal_signed" else 0
        )
        self.recurrent = torch.nn.GRUCell(
            tabular_dimension + graph_dimension, hidden_dimension
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Linear(hidden_dimension, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )

    def _signed_graph_state(
        self,
        events: tuple[torch.Tensor, ...],
        security_count: int,
    ) -> torch.Tensor:
        buyer, seller, security, attributes, value_share = events
        buyer_embedding = self.broker_embedding(buyer)
        seller_embedding = self.broker_embedding(seller)
        zeros = torch.zeros_like(buyer_embedding)
        buy = torch.cat(
            [buyer_embedding, zeros, attributes], dim=1
        )
        sell = torch.cat(
            [zeros, seller_embedding, attributes], dim=1
        )
        messages = self.message(torch.cat([buy, sell], dim=0))
        security = torch.cat([security, security])
        aggregate = messages.new_zeros(
            (security_count, self.message_dimension)
        )
        weights = (
            torch.ones(
                (len(security), 1),
                dtype=messages.dtype,
                device=messages.device,
            )
            if self.graph_aggregation == "mean"
            else torch.cat([value_share, value_share]).reshape(-1, 1).to(
                dtype=messages.dtype
            )
        )
        aggregate.index_add_(0, security, messages * weights)
        denominator = messages.new_zeros((security_count, 1))
        denominator.index_add_(
            0,
            security,
            weights,
        )
        return aggregate / denominator.clamp_min(torch.finfo(messages.dtype).eps)

    def forward_session(
        self,
        tabular: torch.Tensor,
        label_security: torch.Tensor,
        hidden: torch.Tensor,
        events: tuple[torch.Tensor, ...] | None,
        security_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = tabular
        if self.model_type == "temporal_signed":
            if events is None:
                raise ValueError("Signed model requires transaction events")
            graph_state = self._signed_graph_state(events, security_count)
            inputs = torch.cat(
                [tabular, graph_state[label_security]], dim=1
            )
        updated = self.recurrent(inputs, hidden[label_security])
        next_hidden = hidden.index_copy(
            0, label_security, updated.to(hidden.dtype)
        )
        return self.predictor(updated).squeeze(-1), next_hidden
