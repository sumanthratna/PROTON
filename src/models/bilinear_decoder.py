import dgl
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class BilinearDecoder(pl.LightningModule):
    def __init__(
        self,
        num_etypes: int,
        embedding_dim: int,
        edge_signs_enabled: bool = False,
        edge_sign_mapping: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        self.relation_weights = nn.Parameter(torch.Tensor(num_etypes, embedding_dim))
        self.edge_signs_enabled = edge_signs_enabled
        self.edge_sign_mapping = edge_sign_mapping or {}

        nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain("leaky_relu"))

    def _get_edge_sign(self, relation: str) -> int:
        """Get the sign for a relation, handling reverse edges.

        Args:
            relation: The relation name (e.g., 'indication', 'rev_contraindication')

        Returns:
            The sign (+1, -1, or 0) for the relation. Defaults to +1 if not found.
        """
        # Strip 'rev_' prefix for reverse edges to use the same sign as forward edge
        if relation.startswith("rev_"):
            relation = relation[4:]
        return self.edge_sign_mapping.get(relation, 1)

    def add_edge_type_index(self, edge_graph: dgl.DGLHeteroGraph) -> None:
        for edge_index, edge_type in enumerate(edge_graph.canonical_etypes):
            num_edges = edge_graph.num_edges(edge_type)

            edge_graph.edges[edge_type].data["edge_type_index"] = torch.tensor(
                [edge_index] * num_edges, device=self.device
            )

    def _create_decode_fn(self, etype: tuple[str, str, str]):
        """Create a decode function for a specific edge type.

        This is needed because DGL's apply_edges doesn't pass the edge type to the UDF,
        so we create a closure that captures the edge type.
        """

        def decode(edges: dgl.DGLHeteroGraph) -> dict[str, torch.Tensor]:
            src_embeddings = edges.src["node_embedding"]
            dst_embeddings = edges.dst["node_embedding"]

            src_embeddings = F.leaky_relu(src_embeddings)
            dst_embeddings = F.leaky_relu(dst_embeddings)

            edge_type_index = edges.data["edge_type_index"][0]
            decoder = self.relation_weights[edge_type_index]

            score = torch.sum(src_embeddings * decoder * dst_embeddings, dim=1)

            # Apply edge sign if enabled
            if self.edge_signs_enabled:
                relation = etype[1]  # Extract relation name from (src_type, relation, dst_type)
                sign = self._get_edge_sign(relation)
                score = score * sign

            return {"score": score}

        return decode

    def decode(self, edges: dgl.DGLHeteroGraph) -> dict[str, torch.Tensor]:
        """Default decode function (used when edge signs are disabled)."""
        src_embeddings = edges.src["node_embedding"]
        dst_embeddings = edges.dst["node_embedding"]

        src_embeddings = F.leaky_relu(src_embeddings)
        dst_embeddings = F.leaky_relu(dst_embeddings)

        edge_type_index = edges.data["edge_type_index"][0]
        decoder = self.relation_weights[edge_type_index]

        score = torch.sum(src_embeddings * decoder * dst_embeddings, dim=1)

        return {"score": score}

    def compute_score(self, edge_graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        with edge_graph.local_scope():
            nonzero_edge_types = [etype for etype in edge_graph.canonical_etypes if edge_graph.num_edges(etype) != 0]

            for etype in nonzero_edge_types:
                if self.edge_signs_enabled:
                    # Use edge-type-aware decode function to apply signs
                    decode_fn = self._create_decode_fn(etype)
                    edge_graph.apply_edges(decode_fn, etype=etype)
                else:
                    edge_graph.apply_edges(self.decode, etype=etype)

            return edge_graph.edata["score"]

    def forward(
        self,
        subgraph: dgl.DGLHeteroGraph,
        edge_graph: dgl.DGLHeteroGraph,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        subgraph_nodes = subgraph.ndata["node_index"]

        for ntype in edge_graph.ntypes:
            edge_graph_nodes = edge_graph.ndata["node_index"][ntype].unsqueeze(1)

            edge_graph_indices = torch.where(subgraph_nodes == edge_graph_nodes)[1]

            edge_graph.nodes[ntype].data["node_embedding"] = node_embeddings[edge_graph_indices]

        self.add_edge_type_index(edge_graph)

        scores = self.compute_score(edge_graph)

        return scores
