import json
import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from cases.models import FlaggedCase
from core.models import Contract, Supplier
from ingestion.management.commands.import_data import SCREEN_COLUMNS
from ml_pipeline import model_store
from ml_pipeline.data import load_frame
from ml_pipeline.evaluate import choose_threshold, cross_validate
from ml_pipeline.features import EXCLUDED_COLUMNS, FEATURE_COLUMNS, build_features
from ml_pipeline.models import Prediction


def make_dataset(n_cartels=12, per_cartel=40, seed=0):
    """Small synthetic dataset: half positives per cartel, with a real signal in a few screens."""
    rng = np.random.default_rng(seed)
    supplier = Supplier.objects.create(bidder_id="s1")
    contracts = []
    for k in range(n_cartels):
        for i in range(per_cartel):
            positive = i % 2 == 0
            screens = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            screens["lot_bidscount_avg"] += 1.5 * positive
            screens["benfords_market_yearly_avg"] += 1.0 * positive
            contracts.append(Contract(
                tender_id=f"t{k}_{i}", country="country_1", tender_year=2010 + i % 5, supplier=supplier,
                screens=screens, is_cartel=positive, cartel_id=f"case_{k}"))
    Contract.objects.bulk_create(contracts)


class FeatureTests(TestCase):
    def test_features_are_stored_by_ingestion_and_never_excluded(self):
        self.assertEqual(len(FEATURE_COLUMNS), 23)
        self.assertTrue(set(FEATURE_COLUMNS) <= set(SCREEN_COLUMNS))
        self.assertFalse(set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS))

    def test_build_features_rejects_a_missing_column(self):
        make_dataset(n_cartels=1, per_cartel=4)
        frame = load_frame().drop(columns=["singleb_avg"])
        with self.assertRaises(ValueError):
            build_features(frame)


class ValidationTests(TestCase):
    def test_every_cartel_is_in_exactly_one_test_fold(self):
        make_dataset()
        frame = load_frame()
        cv = cross_validate(frame, n_splits=4)
        seen = []
        for _, test in cv.fold_models:
            seen.append(set(frame["cartel_id"].iloc[test]))
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                self.assertFalse(seen[i] & seen[j], "a cartel appears in two test folds (leakage)")
        self.assertFalse(np.isnan(cv.oof_proba).any())


class ThresholdTests(TestCase):
    def test_threshold_flags_the_requested_share(self):
        proba = np.random.default_rng(1).random(1000)
        for rate in (0.05, 0.10, 0.25):
            share = float((proba >= choose_threshold(proba, rate)).mean())
            self.assertAlmostEqual(share, rate, delta=0.01)


class EndToEndTests(TestCase):
    def setUp(self):
        make_dataset()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"MODEL_PATH": str(Path(self.tmp.name) / "classifier.pkl")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_command(self, *args):
        out = StringIO()
        call_command(*args, stdout=out)
        return out.getvalue()

    def test_scoring_requires_a_trained_model(self):
        with self.assertRaises(CommandError):
            self.run_command("score_contracts")

    def test_train_then_score_then_rescore(self):
        self.run_command("train_model", "--folds", "4", "--flag-rate", "0.2")
        self.assertTrue(model_store.artifact_path().exists())
        metrics = json.loads(model_store.metrics_path().read_text())
        for key in ("model_version", "evaluated_on", "baseline_pr_auc", "pr_auc_mean", "pr_auc_std",
                    "precision_mean", "recall_mean", "f1_mean", "confusion_matrix", "flag_rate", "lift",
                    "precision_pooled", "f1_optimal"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["evaluated_on"], "group_kfold_by_cartel_id")
        self.assertGreater(metrics["pr_auc_mean"], metrics["baseline_pr_auc"])  # the synthetic signal is learnable
        self.assertGreater(metrics["lift"], 1.0)

        n = Contract.objects.count()
        self.run_command("score_contracts", "--flag-top", "10")
        self.assertEqual(Prediction.objects.count(), n)
        flagged = Prediction.objects.filter(predicted_label=True).count()
        self.assertAlmostEqual(flagged / n, 0.2, delta=0.03)
        self.assertTrue(all(p.model_version.endswith("-oof") for p in Prediction.objects.all()))
        for prediction in Prediction.objects.all()[:50]:
            self.assertTrue(0.0 <= prediction.risk_score <= 1.0)
            impacts = [abs(f["impact"]) for f in prediction.top_features]
            self.assertTrue(0 < len(impacts) <= 10)
            self.assertEqual(impacts, sorted(impacts, reverse=True))
            self.assertEqual(set(prediction.top_features[0]), {"feature", "value", "impact"})
        self.assertEqual(FlaggedCase.objects.count(), 10)

        # Re-scoring replaces predictions, creates no duplicate cases and keeps analyst decisions
        case = FlaggedCase.objects.first()
        case.status = FlaggedCase.Status.CONFIRMED
        case.comment = "checked"
        case.save()
        self.run_command("score_contracts", "--flag-top", "10")
        self.assertEqual(Prediction.objects.count(), n)
        self.assertEqual(FlaggedCase.objects.count(), 10)
        case.refresh_from_db()
        self.assertEqual((case.status, case.comment), (FlaggedCase.Status.CONFIRMED, "checked"))

        # Final-model mode stores a second, separately versioned set
        self.run_command("score_contracts", "--mode", "final", "--flag-top", "0")
        self.assertEqual(Prediction.objects.count(), 2 * n)
