# %% [markdown]
# # Graph baseline: supplier-buyer network
#
# **Owner:** graph module. Questions this script answers:
# 1. How much of the data can the network use (missing `buyer_id`, the dominant supplier)?
# 2. Do Louvain communities of the supplier-buyer graph recover the real cartels (`cartel_id`)?
# 3. Is that better than chance, and better than simply grouping suppliers by country?
# 4. How sensitive is it to the weighting, the hub-buyer cap, the resolution and the dominant supplier?
# 5. Does it hold on rolling 2-year windows?
#
# Run it cell by cell in VS Code (Run Cell above each `# %%`) or as a normal script, from the repo root
# or from `notebooks/`. Needs: pip install networkx scikit-learn
# `cartel_id` and `country` are used ONLY to validate, never to build the graph.

# %%
import os
import sys
import warnings

import networkx as nx
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)

ROOT = next((p for p in [".", ".."] if os.path.isdir(os.path.join(p, "graph_analysis"))), None)
assert ROOT is not None, "Run this from the repo root or from notebooks/"
sys.path.insert(0, os.path.abspath(ROOT))

from graph_analysis.network import (build_projection, cluster_stats, connected_core, detect_communities,
                                    evaluate_partition, groups_from_labels, random_partition_like,
                                    supplier_labels, windowed_communities)

FILENAME = "GTI_labelled_cartel_data_NOV2023.csv"
CANDIDATES = [os.path.join(ROOT, "data", "raw", FILENAME)]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert DATA_PATH, f"CSV not found. Looked in: {CANDIDATES}"

TEXT = ["tender_id", "bidder_id", "buyer_id", "country", "cartel_id"]
df = pd.read_csv(DATA_PATH, usecols=TEXT + ["tender_year", "is_cartel"],
                 dtype={c: str for c in TEXT}, low_memory=False)
print("Using:", DATA_PATH, "|", df.shape)

# %% [markdown]
# ## 1. What can the network use?

# %%
print("rows:", len(df), "| suppliers:", df.bidder_id.nunique(), "| buyers:", df.buyer_id.nunique())
print(f"rows without buyer_id: {df.buyer_id.isna().mean():.1%}")

print("\nmissing buyer_id by country:")
print(df.groupby("country").buyer_id.apply(lambda s: s.isna().mean()).round(3).to_string())

by_case = df.groupby("cartel_id").agg(n=("buyer_id", "size"), missing=("buyer_id", lambda s: s.isna().mean()))
print("\nlargest cases:")
display(by_case.sort_values("n", ascending=False).head(10).round(3))
print("cases where more than half of the rows have no buyer_id:", int((by_case.missing > 0.5).sum()), "of", len(by_case))

top_supplier = df.bidder_id.value_counts().index[0]
print("\ndominant supplier:", top_supplier, "| rows:", int((df.bidder_id == top_supplier).sum()),
      "| cartels:", df[df.bidder_id == top_supplier].cartel_id.unique().tolist())

truth = supplier_labels(df)
print("suppliers per cartel (median):", truth.value_counts().median(), "| cartels:", truth.nunique())

buyer_degree = df.dropna(subset=["buyer_id"]).groupby("buyer_id").bidder_id.nunique()
print("\nsuppliers per buyer:")
print(buyer_degree.describe().round(1).to_string())

# %% [markdown]
# ## 2. The graph (full period, normalized weights)

# %%
g = build_projection(df, weighting="normalized")
core = connected_core(g)
components = sorted((len(c) for c in nx.connected_components(core)), reverse=True)
print("suppliers in graph:", g.number_of_nodes(), "| isolated:", g.number_of_nodes() - core.number_of_nodes(),
      "| edges:", core.number_of_edges())
print("connected components:", len(components), "| largest sizes:", components[:8])

# %% [markdown]
# ## 3. Do the communities recover the cartels?
#
# NMI / ARI / purity compare the communities with the real cartels (1.0 = perfect, ~0 = chance).
# Purity is flattering when there are many small communities, so read it next to the baselines:
# **random** (same community sizes, random members) and **by country** (suppliers grouped by their country).

# %%
communities = detect_communities(g)
country_of = df.groupby("bidder_id").country.agg(lambda s: s.value_counts().idxmax())
members = [n for c in communities for n in c]

random_runs = pd.DataFrame([evaluate_partition(random_partition_like(communities, s), truth) for s in range(20)])
table = pd.DataFrame([
    {"partition": "louvain (supplier-buyer graph)", **evaluate_partition(communities, truth)},
    {"partition": "random, same sizes (mean of 20)", **random_runs.mean().to_dict()},
    {"partition": "by country", **evaluate_partition(groups_from_labels(country_of.loc[members]), truth)},
])
display(table.round(3))
print("community sizes (largest 10):", sorted((len(c) for c in communities), reverse=True)[:10])

# %% [markdown]
# ## 4. Sensitivity
# Weighting, hub-buyer cap, dominant supplier and Louvain resolution (higher = more, smaller communities).

# %%
rows = []
for weighting in ("shared", "normalized"):
    for hub_cap in (None, 30):
        for excluded in (False, True):
            graph = build_projection(df, weighting=weighting, max_buyer_degree=hub_cap,
                                     exclude_suppliers=[top_supplier] if excluded else ())
            for resolution in (0.5, 1.0, 2.0):
                result = evaluate_partition(detect_communities(graph, resolution), truth)
                rows.append({"weighting": weighting, "hub_cap": hub_cap, "dominant_excluded": excluded,
                             "resolution": resolution, **result})
sensitivity = pd.DataFrame(rows)
display(sensitivity.round(3))

# %% [markdown]
# ## 5. Rolling 2-year windows
# Pick the configuration after reading the table above (defaults below), then compare every window with its random baseline.

# %%
CONFIG = dict(weighting="normalized", exclude_suppliers=[top_supplier])
RESOLUTION = 1.0

windows = windowed_communities(df, width=2, step=1, resolution=RESOLUTION, **CONFIG)
rows = []
for w in windows:
    r = evaluate_partition(w["communities"], truth)
    base = pd.DataFrame([evaluate_partition(random_partition_like(w["communities"], s), truth) for s in range(10)]).mean()
    rows.append({"window": f"{w['year_from']}-{w['year_to']}", "n_suppliers": r["n_suppliers"],
                 "n_communities": r["n_communities"], "nmi": r["nmi"], "nmi_random": base["nmi"],
                 "ari": r["ari"], "ari_random": base["ari"], "purity": r["purity"], "purity_random": base["purity"]})
by_window = pd.DataFrame(rows)
display(by_window.round(3))

valid = by_window.dropna(subset=["nmi"])
if len(valid):
    weights = valid.n_suppliers
    print("size-weighted means:", {c: round(float(np.average(valid[c], weights=weights)), 3)
                                   for c in ["nmi", "nmi_random", "ari", "ari_random", "purity", "purity_random"]})

# %% [markdown]
# ## 6. What do the biggest communities look like?
# `top_cartel_share` = share of the community's suppliers that belong to its most common real cartel.

# %%
rows = []
for c in sorted(communities, key=len, reverse=True)[:10]:
    labels = truth.loc[list(c)].value_counts()
    rows.append({**cluster_stats(g, c, df), "top_cartel": labels.index[0],
                 "top_cartel_share": labels.iloc[0] / len(c), "n_cartels": len(labels)})
display(pd.DataFrame(rows).round(3))

# %% [markdown]
# ## 7. Notes for `docs/eda_findings.md`
# - Missing `buyer_id`: overall ..., concentrated in ... (cases / countries)
# - Dominant supplier: ..., effect of excluding it: ...
# - Louvain vs random vs by-country (NMI / ARI / purity): ...
# - Best configuration (weighting, hub cap, resolution): ...
# - Rolling windows: ...
# - Biggest communities: do they match one cartel? ...
# - Verdict: does the supplier-buyer network recover cartel structure, and how well compared with the baselines? ...