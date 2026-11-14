"""Find communities in the supplier-buyer graph and store them (clusters, members, edges, scores).

    python manage.py compute_graph
    python manage.py compute_graph --window 2 --step 2 --resolution 1.0 --min-size 3 --flag-top 20

Choices backed by notebooks/graph_baseline.py: normalized weights (hub buyers count less), the dominant
supplier excluded, Louvain resolution 1.0, non-overlapping 2-year windows.
`cartel_id` is used only to validate (metrics.json and the printed correlations), never to build clusters.
"""
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from cases.models import FlaggedCase
from core.models import Contract
from graph_analysis import metrics_store
from graph_analysis.models import Cluster, ClusterEdge, ClusterMember
from graph_analysis.network import (cluster_stats, evaluate_partition, random_partition_like,
                                    supplier_labels, windowed_communities)
from ml_pipeline.models import Prediction


def _clean(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


class Command(BaseCommand):
    help = "Detect supplier communities per time window and store clusters, members, edges and scores."

    def add_arguments(self, parser):
        parser.add_argument("--window", type=int, default=2, help="Window length in years")
        parser.add_argument("--step", type=int, default=2, help="Years between window starts (= window: no overlap)")
        parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution (higher = smaller clusters)")
        parser.add_argument("--min-size", type=int, default=3, help="Store only clusters with at least this many suppliers")
        parser.add_argument("--dominant-share", type=float, default=0.10,
                            help="Exclude suppliers holding more than this share of all contracts (0 = keep all)")
        parser.add_argument("--flag-top", type=int, default=20, help="Open a pending case for the N highest-scoring clusters")
        parser.add_argument("--model-version", default=None, help="Predictions to use (default: the latest '-oof' set)")
        parser.add_argument("--force", action="store_true",
                            help="Recompute even if analysts already reviewed cluster cases (their decisions are lost)")

    def handle(self, *args, **options):
        rows = list(Contract.objects.values("id", "supplier_id", "buyer_id", "tender_year", "cartel_id", "is_cartel"))
        if not rows:
            raise CommandError("No contracts in the database. Run import_data first.")
        reviewed = FlaggedCase.objects.filter(source=FlaggedCase.Source.CLUSTER).exclude(status=FlaggedCase.Status.PENDING)
        if reviewed.exists() and not options["force"]:
            raise CommandError("Analysts already reviewed cluster cases; recomputing would delete them. Use --force to override.")

        # Nodes are Supplier primary keys, buyers are Buyer primary keys (NaN = unknown buyer)
        frame = pd.DataFrame(rows).rename(columns={"id": "contract_id", "supplier_id": "bidder_id"})

        counts = frame["bidder_id"].value_counts()
        excluded = [int(s) for s in counts[counts / len(frame) > options["dominant_share"]].index] \
            if options["dominant_share"] > 0 else []
        if excluded:
            self.stdout.write(f"Excluding dominant supplier(s) {excluded} (more than {options['dominant_share']:.0%} of all contracts).")

        risk, risk_version = self.load_risk(options["model_version"])
        truth = supplier_labels(frame)
        windows = windowed_communities(frame, width=options["window"], step=options["step"],
                                       resolution=options["resolution"], weighting="normalized",
                                       exclude_suppliers=excluded)

        created, window_metrics = [], []
        with transaction.atomic():
            Cluster.objects.all().delete()   # cascades to members, edges and their cases
            for w in windows:
                metrics = evaluate_partition(w["communities"], truth)
                baseline = pd.DataFrame([evaluate_partition(random_partition_like(w["communities"], s), truth)
                                         for s in range(10)]).mean() if w["communities"] else None
                window_metrics.append({
                    "window": f"{w['year_from']}-{w['year_to']}", **metrics,
                    **({f"{k}_random": float(baseline[k]) for k in ("nmi", "ari", "purity")} if baseline is not None else {})})
                for community in w["communities"]:
                    if len(community) >= options["min_size"]:
                        created.append(self.store_cluster(w, community, risk, risk_version, options, excluded))

            n_cases = self.flag_clusters(created, options["flag_top"])
            metrics_store.save(self.summarize(window_metrics, len(created), bool(excluded), options))

        self.stdout.write(self.style.SUCCESS(f"Stored {len(created)} clusters from {len(windows)} windows."))
        self.stdout.write(f"  mean model risk from: {risk_version or 'no predictions found'} | cases opened: {n_cases}")
        self.report_validation(created)

    @staticmethod
    def load_risk(version):
        qs = Prediction.objects.all()
        if version is None:
            latest = qs.filter(model_version__endswith="-oof").order_by("-created_at").first()
            version = latest.model_version if latest else None
        if version is None:
            return pd.Series(dtype=float), None
        return pd.Series(dict(qs.filter(model_version=version).values_list("contract_id", "risk_score"))), version

    def store_cluster(self, w, community, risk, risk_version, options, excluded):
        data = w["data"]
        rows = data[data["bidder_id"].isin(community)]
        stats = cluster_stats(w["graph"], community, data)
        mean_risk = _clean(risk.reindex(rows["contract_id"]).mean()) if len(risk) else None
        parts = [stats["density"], stats["buyer_hhi"]] + ([mean_risk] if mean_risk is not None else [])
        score = float(np.mean(parts))

        cluster = Cluster.objects.create(
            year_from=w["year_from"], year_to=w["year_to"], size=stats["size"], density=stats["density"],
            n_contracts=stats["n_contracts"], n_buyers=stats["n_buyers"], buyer_concentration=stats["buyer_hhi"],
            mean_model_risk=mean_risk, suspicion_score=score,
            config={"weighting": "normalized", "resolution": options["resolution"], "window": options["window"],
                    "step": options["step"], "excluded_suppliers": excluded, "risk_model_version": risk_version,
                    "score_components": ["density", "buyer_concentration"] + (["mean_model_risk"] if mean_risk is not None else [])})

        per_supplier = rows.groupby("bidder_id").agg(contracts=("contract_id", "size"), buyers=("buyer_id", "nunique"))
        ClusterMember.objects.bulk_create([
            ClusterMember(cluster=cluster, supplier_id=int(s), contract_count=int(per_supplier.loc[s, "contracts"]),
                          n_buyers=int(per_supplier.loc[s, "buyers"]))
            for s in community])
        ClusterEdge.objects.bulk_create([
            ClusterEdge(cluster=cluster, source_id=int(u), target_id=int(v), weight=float(d["shared"]))   # weight = shared buyers
            for u, v, d in w["graph"].subgraph(community).edges(data=True)])

        pos_rate = float(rows["is_cartel"].mean())   # validation only
        return {"cluster": cluster, "pos_rate": pos_rate}

    @staticmethod
    def flag_clusters(created, top_n):
        if top_n <= 0:
            return 0
        best = sorted(created, key=lambda c: c["cluster"].suspicion_score, reverse=True)[:top_n]
        FlaggedCase.objects.bulk_create([
            FlaggedCase(source=FlaggedCase.Source.CLUSTER, cluster=c["cluster"], score=c["cluster"].suspicion_score)
            for c in best])
        return len(best)

    @staticmethod
    def summarize(window_metrics, n_clusters, dominant_excluded, options):
        valid = [m for m in window_metrics if m.get("nmi") is not None and not np.isnan(m["nmi"])]
        weights = [m["n_suppliers"] for m in valid]

        def mean(key):
            values = [m[key] for m in valid if key in m]
            return float(np.average(values, weights=weights[:len(values)])) if values else None

        return {
            "window": f"{options['window']}-year windows",
            "n_windows": len(window_metrics), "n_clusters": n_clusters,
            "nmi": mean("nmi"), "ari": mean("ari"), "purity": mean("purity"),
            "nmi_random": mean("nmi_random"), "ari_random": mean("ari_random"), "purity_random": mean("purity_random"),
            "dominant_supplier_excluded": dominant_excluded,
            "windows": [{k: _clean(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}
                        for m in window_metrics],
        }

    def report_validation(self, created):
        if len(created) < 3:
            return
        df = pd.DataFrame([{"score": c["cluster"].suspicion_score, "density": c["cluster"].density,
                            "buyer_concentration": c["cluster"].buyer_concentration,
                            "mean_model_risk": c["cluster"].mean_model_risk, "size": c["cluster"].size,
                            "pos_rate": c["pos_rate"]} for c in created])
        self.stdout.write("  rank correlation with the share of cartel-labelled contracts (validation only):")
        for column in ["score", "density", "buyer_concentration", "mean_model_risk", "size"]:
            if df[column].notna().sum() >= 3:
                self.stdout.write(f"    {column:20s} {df[column].corr(df['pos_rate'], method='spearman'):+.2f}")