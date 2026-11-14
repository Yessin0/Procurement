"""Supplier-buyer network analysis. Pure functions (no Django): DataFrame in, results out.

The dataset only records winners, so suppliers are linked through the BUYERS they win contracts
from, not through co-bidding. Expected columns: bidder_id, buyer_id (NaN allowed), tender_year.
`cartel_id` / `country` are used only to validate the communities, never to build them.
"""
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def build_projection(df, weighting="normalized", max_buyer_degree=None, exclude_suppliers=()):
    """Supplier-supplier graph: two suppliers are linked when they won contracts from the same buyer.

    weighting="shared":     weight = number of shared buyers
    weighting="normalized": every shared buyer counts 1/(d-1), d = number of suppliers of that buyer,
                            so hub buyers (who work with many suppliers) create weaker links
    max_buyer_degree: ignore buyers with more than this many suppliers (None = keep all)
    Every edge also stores `shared` (number of shared buyers). Rows without buyer_id are ignored.
    """
    if weighting not in ("normalized", "shared"):
        raise ValueError("weighting must be 'normalized' or 'shared'")
    used = df.dropna(subset=["buyer_id"])
    used = used[~used["bidder_id"].isin(set(exclude_suppliers))]
    pairs = used[["bidder_id", "buyer_id"]].drop_duplicates()

    graph = nx.Graph()
    graph.add_nodes_from(pairs["bidder_id"].unique())
    acc = {}
    for _, suppliers in pairs.groupby("buyer_id")["bidder_id"]:
        suppliers = sorted(suppliers)
        d = len(suppliers)
        if d < 2 or (max_buyer_degree and d > max_buyer_degree):
            continue
        weight = 1.0 if weighting == "shared" else 1.0 / (d - 1)
        for a, b in combinations(suppliers, 2):
            entry = acc.setdefault((a, b), [0.0, 0])
            entry[0] += weight
            entry[1] += 1
    graph.add_edges_from((a, b, {"weight": w, "shared": n}) for (a, b), (w, n) in acc.items())
    return graph


def connected_core(graph):
    """The graph without isolated suppliers (as singleton 'communities' they would inflate purity)."""
    return graph.subgraph([n for n, degree in graph.degree() if degree > 0]).copy()


def detect_communities(graph, resolution=1.0, seed=42):
    """Louvain communities of the connected part of the graph (list of sets of suppliers)."""
    core = connected_core(graph)
    if core.number_of_nodes() == 0:
        return []
    return [set(c) for c in louvain_communities(core, weight="weight", resolution=resolution, seed=seed)]


def supplier_labels(df, column="cartel_id"):
    """Majority label per supplier (996 of the 1,001 suppliers belong to a single cartel)."""
    return df.groupby("bidder_id")[column].agg(lambda s: s.value_counts().idxmax())


def groups_from_labels(labels):
    """A Series node -> label becomes a list of sets (one per label)."""
    return [set(index) for index in labels.groupby(labels).groups.values()]


def random_partition_like(communities, seed=0):
    """Same community sizes, random members: the 'no structure' baseline."""
    rng = np.random.default_rng(seed)
    nodes = [n for c in communities for n in c]
    rng.shuffle(nodes)
    result, start = [], 0
    for c in communities:
        result.append(set(nodes[start:start + len(c)]))
        start += len(c)
    return result


def evaluate_partition(communities, truth):
    """NMI, ARI and purity of a partition against the true labels (suppliers present in both)."""
    node_to_community = {n: i for i, c in enumerate(communities) for n in c}
    nodes = [n for n in node_to_community if n in truth.index]
    if len(nodes) < 2:
        return {"n_suppliers": len(nodes), "n_communities": len(communities),
                "nmi": np.nan, "ari": np.nan, "purity": np.nan}
    predicted = [node_to_community[n] for n in nodes]
    actual = truth.loc[nodes].tolist()
    table = pd.crosstab(pd.Series(predicted, name="p"), pd.Series(actual, name="t"))
    return {
        "n_suppliers": len(nodes),
        "n_communities": len(set(predicted)),
        "nmi": float(normalized_mutual_info_score(actual, predicted)),
        "ari": float(adjusted_rand_score(actual, predicted)),
        "purity": float(table.max(axis=1).sum() / table.values.sum()),
    }


def windowed_communities(df, width=2, step=1, resolution=1.0, **projection_kwargs):
    """One graph + Louvain run per rolling window of `width` years (cartels evolve over time)."""
    first, last = int(df["tender_year"].min()), int(df["tender_year"].max())
    results = []
    for start in range(first, last - width + 2, step):
        end = start + width - 1
        window = df[(df["tender_year"] >= start) & (df["tender_year"] <= end)]
        graph = build_projection(window, **projection_kwargs)
        results.append({"year_from": start, "year_to": end, "graph": graph, "data": window,
                        "communities": detect_communities(graph, resolution)})
    return results


def cluster_stats(graph, community, data):
    """Descriptive numbers of one community; `data` holds the contracts of the same period."""
    rows = data[data["bidder_id"].isin(community)]
    known = rows.dropna(subset=["buyer_id"])
    shares = known["buyer_id"].value_counts(normalize=True)
    return {
        "size": len(community),
        "density": float(nx.density(graph.subgraph(community))) if len(community) > 1 else 0.0,
        "n_contracts": int(len(rows)),
        "n_buyers": int(known["buyer_id"].nunique()),
        "buyer_hhi": float((shares ** 2).sum()) if len(shares) else 0.0,  # 1 = all contracts from one buyer
    }