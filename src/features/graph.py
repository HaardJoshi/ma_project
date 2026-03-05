"""
Block C — Graph / SPLC topology features.

Computes network metrics and optional GraphSAGE embeddings from the
supplier/customer/competitor graph built from Bloomberg SPLC data.

TODO
----
- SPLC data loader (acquirer → suppliers, customers, competitors)
- Graph construction (nodes = firms, typed edges)
- Classical metrics: degree, betweenness centrality, clustering coefficient
- Optional: unsupervised GraphSAGE for 16–32 dim embedding h_G
- Graph snapshot at t-Δ (pre-announcement)
"""

from typing import Optional


def compute_graph_metrics(
    acquirer_ticker: str,
    graph: Optional[object] = None,
) -> dict:
    """
    Compute classical network metrics for an acquirer node.

    Parameters
    ----------
    acquirer_ticker : str
        Bloomberg ticker of the acquiring firm.
    graph : optional
        Pre-built NetworkX or PyG graph. If None, loads default.

    Returns
    -------
    dict
        Keys: degree, in_degree, out_degree, betweenness_centrality,
        clustering_coefficient, supply_concentration.

    TODO: Implement when SPLC data pipeline is ready.
    """
    raise NotImplementedError(
        "Graph metrics not yet implemented. "
        "Requires: SPLC data download and graph construction."
    )


def compute_graphsage_embedding(
    acquirer_ticker: str,
    graph: Optional[object] = None,
    embed_dim: int = 32,
) -> list[float]:
    """
    Compute GraphSAGE embedding h_G for an acquirer node.

    Parameters
    ----------
    acquirer_ticker : str
        Bloomberg ticker.
    graph : optional
        Pre-built PyG HeteroData graph.
    embed_dim : int
        Output embedding dimensionality.

    Returns
    -------
    list[float]
        GraphSAGE embedding vector.

    TODO: Implement when SPLC data pipeline is ready.
    """
    raise NotImplementedError("GraphSAGE embedding not yet implemented.")
