"""Write a risk score + SHAP explanation for every contract to the database.

    python manage.py score_contracts                  # honest out-of-fold scores (default)
    python manage.py score_contracts --mode final     # final model (use for NEW contracts)
    python manage.py score_contracts --flag-top 300   # open a case for the 300 riskiest contracts
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from cases.models import FlaggedCase
from ml_pipeline import model_store
from ml_pipeline.data import load_frame
from ml_pipeline.evaluate import cross_validate
from ml_pipeline.models import Prediction
from ml_pipeline.scoring import score_final, score_out_of_fold


class Command(BaseCommand):
    help = "Score contracts with the trained model and store predictions (and flagged cases)."

    def add_arguments(self, parser):
        parser.add_argument("--mode", choices=["oof", "final"], default="oof",
                            help="oof: every contract is scored by a model that never saw its cartel (honest, for the "
                                 "labelled dataset). final: score with the final model (in-sample on the dataset).")
        parser.add_argument("--flag-top", type=int, default=300,
                            help="Open a pending case for the N highest-risk contracts (0 = no cases).")
        parser.add_argument("--max-per-supplier", type=int, default=3,
                            help="Flag at most this many contracts per supplier (0 = no limit). Contracts of one supplier "
                                 "share most screens, get near-identical scores and would otherwise fill the whole list.")
        parser.add_argument("--batch-size", type=int, default=1000)

    def handle(self, *args, **options):
        try:
            art = model_store.load()
        except FileNotFoundError as exc:
            raise CommandError(str(exc))
        frame = load_frame()
        if frame.empty:
            raise CommandError("No contracts in the database. Run import_data first.")

        threshold = art["threshold"]
        if options["mode"] == "oof":
            version = f"{art['model_version']}-oof"
            self.stdout.write("Re-running the group split to score each contract out-of-fold...")
            scored = score_out_of_fold(frame, cross_validate(frame), threshold, version)
        else:
            version = art["model_version"]
            scored = score_final(frame, art["model"], threshold, version)

        with transaction.atomic():
            Prediction.objects.filter(model_version=version).delete()
            Prediction.objects.bulk_create(
                [Prediction(contract_id=int(r.contract_id), model_version=version, risk_score=float(r.risk_score),
                            predicted_label=bool(r.predicted_label), threshold=float(r.threshold),
                            top_features=r.top_features)
                 for r in scored.itertuples()],
                batch_size=options["batch_size"])
            supplier_of = dict(zip(frame["contract_id"], frame["supplier_id"]))
            n_new, n_updated, n_removed = self.flag_cases(scored, options["flag_top"], options["max_per_supplier"], supplier_of)

        self.stdout.write(self.style.SUCCESS(f"Stored {len(scored)} predictions ({version})."))
        self.stdout.write(f"  predicted suspicious: {int(scored.predicted_label.sum())} | threshold {threshold:.3f}")
        if options["flag_top"]:
            self.stdout.write(f"  cases: {n_new} new, {n_updated} pending re-scored, {n_removed} pending dropped "
                              f"(no longer in the list). Reviewed cases are never touched.")

    @staticmethod
    def flag_cases(scored, top_n, max_per_supplier, supplier_of):
        if top_n <= 0:
            return 0, 0, 0
        ranked = scored.sort_values("risk_score", ascending=False)
        if max_per_supplier > 0:
            ranked = ranked.assign(supplier=ranked["contract_id"].map(supplier_of))
            ranked = ranked.groupby("supplier", sort=False).head(max_per_supplier)
        top = ranked.head(top_n)
        score_by_contract = {int(r.contract_id): float(r.risk_score) for r in top.itertuples()}

        existing = list(FlaggedCase.objects.filter(source=FlaggedCase.Source.MODEL,
                                                   contract_id__in=score_by_contract))
        known = {case.contract_id for case in existing}
        pending = [case for case in existing if case.status == FlaggedCase.Status.PENDING]
        for case in pending:
            case.score = score_by_contract[case.contract_id]
        FlaggedCase.objects.bulk_update(pending, ["score"])

        new_cases = [FlaggedCase(source=FlaggedCase.Source.MODEL, contract_id=cid, score=score)
                     for cid, score in score_by_contract.items() if cid not in known]
        FlaggedCase.objects.bulk_create(new_cases)

        # Unreviewed cases that dropped out of the list are removed; analyst decisions are never touched
        stale = FlaggedCase.objects.filter(source=FlaggedCase.Source.MODEL, status=FlaggedCase.Status.PENDING) \
                                   .exclude(contract_id__in=score_by_contract)
        n_removed = stale.count()
        stale.delete()
        return len(new_cases), len(pending), n_removed