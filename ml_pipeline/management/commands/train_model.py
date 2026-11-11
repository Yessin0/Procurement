"""Validate (group split by cartel), choose a threshold, fit the final model and save it.

    python manage.py train_model
"""
from django.core.management.base import BaseCommand, CommandError

from ml_pipeline import model_store
from ml_pipeline.data import load_frame
from ml_pipeline.evaluate import N_FOLDS, choose_threshold, cross_validate, summarize
from ml_pipeline.features import FEATURE_COLUMNS
from ml_pipeline.train import fit_final, new_model_version


class Command(BaseCommand):
    help = "Cross-validate by cartel case, fit the final model and save it with its metrics."

    def add_arguments(self, parser):
        parser.add_argument("--folds", type=int, default=N_FOLDS)
        parser.add_argument("--flag-rate", type=float, default=0.10,
                            help="Share of contracts flagged as suspicious (the riskiest ones). Default 0.10.")

    def handle(self, *args, **options):
        frame = load_frame()
        if frame.empty:
            raise CommandError("No contracts in the database. Run import_data first.")
        if frame["cartel_id"].nunique() < options["folds"]:
            raise CommandError(f"Need at least {options['folds']} cartel cases for {options['folds']} folds.")

        self.stdout.write(f"Cross-validating on {len(frame)} contracts ({frame['cartel_id'].nunique()} cartel cases)...")
        cv = cross_validate(frame, n_splits=options["folds"])
        y = frame["is_cartel"].astype(int).to_numpy()
        threshold = choose_threshold(cv.oof_proba, options["flag_rate"])
        version = new_model_version()
        metrics = summarize(frame, cv, threshold, version, options["flag_rate"])

        self.stdout.write("Fitting the final model on all contracts...")
        model = fit_final(frame)
        model_store.save(model, threshold, version, FEATURE_COLUMNS, metrics)

        self.stdout.write(self.style.SUCCESS(f"Saved {version} to {model_store.artifact_path()}"))
        self.stdout.write(f"  PR-AUC {metrics['pr_auc_mean']:.3f} +/- {metrics['pr_auc_std']:.3f} "
                          f"(random baseline {metrics['baseline_pr_auc']:.3f})")
        self.stdout.write(f"  operating point: flag the riskiest {options['flag_rate']:.0%} (score >= {threshold:.3f})")
        self.stdout.write(f"    precision {metrics['precision_pooled']:.3f} (lift {metrics['lift']:.2f}x over the "
                          f"{metrics['baseline_pr_auc']:.3f} base rate), recall {metrics['recall_pooled']:.3f}")
        best = metrics["f1_optimal"]
        self.stdout.write(f"  for reference, the F1-optimal threshold ({best['threshold']:.3f}) would flag "
                          f"{best['flagged_share']:.0%} of contracts")
