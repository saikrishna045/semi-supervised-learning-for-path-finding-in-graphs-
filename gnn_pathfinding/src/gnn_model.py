
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEPathModel(nn.Module):
    """
    GraphSAGE model for learning node representations
    for source-to-destination pathfinding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)

        self.dropout = dropout

        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index):
        """Generate an embedding for each graph node."""

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x

    def score_candidates(
        self,
        embeddings,
        source,
        destination,
        candidates,
    ):
        """Score possible next-hop nodes."""

        source_embedding = embeddings[source].expand(
            candidates.numel(), -1
        )

        destination_embedding = embeddings[destination].expand(
            candidates.numel(), -1
        )

        candidate_embeddings = embeddings[candidates]

        combined = torch.cat(
            [
                source_embedding,
                destination_embedding,
                candidate_embeddings,
            ],
            dim=-1,
        )

        return self.scorer(combined).squeeze(-1)

    def forward(self, data):
        """Return node embeddings from a PyG Data object."""

        return self.encode(data.x, data.edge_index)

