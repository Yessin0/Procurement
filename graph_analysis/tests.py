import networkx as nx
import pandas as pd
from django.test import SimpleTestCase

from graph_analysis.network import (build_projection, cluster_stats, connected_core, detect_communities,
                                    evaluate_partition, groups_from_labels, random_partition_like,
                                    windowed_communities)


def frame(rows):
    return pd.DataFrame(rows, columns=["bidder_id", "buyer_id", "tender_year"])


class ProjectionTests(SimpleTestCase):
    def setUp(self):
        self.df = frame([("s1", "b1", 2010), ("s2", "b1", 2010), ("s3", "b1", 2010),
                         ("s1", "b2", 2011), ("s2", "b2", 2011)])

    def test_suppliers_sharing_buyers_are_linked(self):
        shared = build_projection(self.df, weighting="shared")
        self.assertEqual(shared["s1"]["s2"]["weight"], 2)      # b1 and b2
        self.assertEqual(shared["s1"]["s3"]["weight"], 1)      # b1 only
        self.assertEqual(shared["s1"]["s2"]["shared"], 2)

    def test_normalized_weights_discount_hub_buyers(self):
        normalized = build_projection(self.df, weighting="normalized")
        self.assertAlmostEqual(normalized["s1"]["s2"]["weight"], 0.5 + 1.0)   # b1 has 3 suppliers, b2 has 2
        self.assertEqual(normalized["s1"]["s2"]["shared"], 2)

    def test_rows_without_buyer_are_ignored(self):
        df = frame([("s1", "b1", 2010), ("s2", "b1", 2010), ("s9", None, 2010)])
        self.assertNotIn("s9", build_projection(df))

    def test_hub_cap_and_exclusion(self):
        self.assertEqual(build_projection(self.df, max_buyer_degree=2).number_of_edges(), 1)   # only b2 counts
        self.assertNotIn("s1", build_projection(self.df, exclude_suppliers=["s1"]))

    def test_unknown_weighting_is_rejected(self):
        with self.assertRaises(ValueError):
            build_projection(self.df, weighting="bogus")


class CommunityTests(SimpleTestCase):
    def two_groups(self):
        rows = []
        for group, buyers in (("a", ["ba1", "ba2"]), ("b", ["bb1", "bb2"])):
            for s in range(4):
                for buyer in buyers:
                    rows.append((f"{group}{s}", buyer, 2010))
        rows.append(("lonely", "b_only_lonely", 2010))
        return frame(rows)

    def test_louvain_recovers_obvious_groups_and_ignores_isolated_suppliers(self):
        df = self.two_groups()
        graph = build_projection(df)
        self.assertIn("lonely", graph)
        self.assertNotIn("lonely", connected_core(graph))
        communities = detect_communities(graph)
        self.assertEqual(sorted(len(c) for c in communities), [4, 4])

        truth = pd.Series({f"{g}{s}": g for g in "ab" for s in range(4)})
        result = evaluate_partition(communities, truth)
        self.assertAlmostEqual(result["nmi"], 1.0)
        self.assertAlmostEqual(result["ari"], 1.0)
        self.assertAlmostEqual(result["purity"], 1.0)

    def test_random_partition_keeps_sizes_and_members(self):
        communities = [{"a", "b", "c"}, {"d", "e"}]
        shuffled = random_partition_like(communities, seed=3)
        self.assertEqual([len(c) for c in shuffled], [3, 2])
        self.assertEqual(set().union(*shuffled), {"a", "b", "c", "d", "e"})

    def test_groups_from_labels(self):
        groups = groups_from_labels(pd.Series({"a": "x", "b": "x", "c": "y"}))
        self.assertCountEqual([len(g) for g in groups], [2, 1])


class WindowAndStatsTests(SimpleTestCase):
    def test_rolling_windows(self):
        df = frame([("s1", "b1", 2010), ("s2", "b1", 2011), ("s1", "b2", 2012), ("s2", "b2", 2013)])
        windows = windowed_communities(df, width=2, step=1)
        self.assertEqual([(w["year_from"], w["year_to"]) for w in windows],
                         [(2010, 2011), (2011, 2012), (2012, 2013)])

    def test_cluster_stats(self):
        df = frame([("s1", "b1", 2010), ("s2", "b1", 2010), ("s1", "b1", 2010), ("s2", "b2", 2010)])
        graph = build_projection(df)
        stats = cluster_stats(graph, {"s1", "s2"}, df)
        self.assertEqual((stats["size"], stats["n_contracts"], stats["n_buyers"]), (2, 4, 2))
        self.assertEqual(stats["density"], 1.0)
        self.assertAlmostEqual(stats["buyer_hhi"], 0.75 ** 2 + 0.25 ** 2)

# --- compute_graph (needs the database) -------------------------------------------------------------------
import json
import tempfile
from io import StringIO
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from cases.models import FlaggedCase
from core.models import Buyer, Contract, Supplier
from graph_analysis import metrics_store
from graph_analysis.models import Cluster, ClusterEdge, ClusterMember
from ml_pipeline.models import Prediction


class ComputeGraphTests(TestCase):
    """Two groups of 5 suppliers (each sharing 2 buyers) plus one dominant supplier bridging both groups."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        override = override_settings(BASE_DIR=Path(tmp.name))
        override.enable()
        self.addCleanup(override.disable)

        buyers = {name: Buyer.objects.create(buyer_id=name) for name in ["ba1", "ba2", "bb1", "bb2"]}
        contracts = []

        def add(supplier, buyer, n, group):
            for i in range(n):
                contracts.append(Contract(tender_id=f"t{len(contracts)}", country="country_1", tender_year=2010 + i % 2,
                                          buyer=buyers[buyer], supplier=supplier, is_cartel=group == "a",
                                          cartel_id=f"case_{group}", screens={}))

        for group in "ab":
            for s in range(5):
                supplier = Supplier.objects.create(bidder_id=f"{group}{s}")
                for buyer in (f"b{group}1", f"b{group}2"):
                    add(supplier, buyer, 2, group)
        dominant = Supplier.objects.create(bidder_id="dom")
        for buyer in ("ba1", "bb1"):
            add(dominant, buyer, 30, "a")
        Contract.objects.bulk_create(contracts)

    def run_command(self, *args):
        out = StringIO()
        call_command("compute_graph", *args, stdout=out)
        return out.getvalue()

    def test_stores_clusters_members_edges_scores_and_metrics(self):
        self.run_command("--flag-top", "5")
        self.assertEqual(Cluster.objects.count(), 2)
        for cluster in Cluster.objects.all():
            self.assertEqual(cluster.size, 5)
            self.assertEqual(cluster.members.count(), 5)
            self.assertEqual(cluster.edges.count(), 10)                       # complete graph on 5 suppliers
            self.assertTrue(all(e.weight == 2 for e in cluster.edges.all()))  # weight = shared buyers
            self.assertTrue(0.0 <= cluster.suspicion_score <= 1.0)
            self.assertIsNone(cluster.mean_model_risk)                       # no predictions yet
            self.assertEqual(len(cluster.config["excluded_suppliers"]), 1)    # the dominant supplier
        self.assertFalse(ClusterMember.objects.filter(supplier__bidder_id="dom").exists())
        self.assertEqual(FlaggedCase.objects.filter(source="cluster").count(), 2)

        metrics = metrics_store.load_metrics()
        self.assertTrue(metrics["dominant_supplier_excluded"])
        self.assertAlmostEqual(metrics["nmi"], 1.0)

    def test_model_risk_feeds_the_cluster_score(self):
        for contract in Contract.objects.all():
            Prediction.objects.create(contract=contract, model_version="v-oof", threshold=0.5,
                                      risk_score=0.9 if contract.cartel_id == "case_a" else 0.1,
                                      predicted_label=contract.cartel_id == "case_a")
        self.run_command()
        by_risk = sorted(Cluster.objects.all(), key=lambda c: c.mean_model_risk)
        self.assertAlmostEqual(by_risk[0].mean_model_risk, 0.1)
        self.assertAlmostEqual(by_risk[1].mean_model_risk, 0.9)
        self.assertGreater(by_risk[1].suspicion_score, by_risk[0].suspicion_score)

    def test_dominant_supplier_can_be_kept(self):
        self.run_command("--dominant-share", "0")
        self.assertTrue(ClusterMember.objects.filter(supplier__bidder_id="dom").exists())
        self.assertFalse(metrics_store.load_metrics()["dominant_supplier_excluded"])

    def test_recompute_replaces_clusters_and_protects_reviewed_cases(self):
        self.run_command()
        self.run_command()
        self.assertEqual(Cluster.objects.count(), 2)

        case = FlaggedCase.objects.filter(source="cluster").first()
        case.status = FlaggedCase.Status.CONFIRMED
        case.save()
        with self.assertRaises(CommandError):
            self.run_command()
        self.run_command("--force")
        self.assertEqual(Cluster.objects.count(), 2)