"""Import the labelled cartel dataset into PostgreSQL.

Usage:
    python manage.py import_data --source data/raw/GTI_labelled_cartel_data_NOV2023.csv
    python manage.py import_data --source data/raw/<file>.csv --limit 200      # random sample
    python manage.py import_data --source data/raw/<file>.csv --clear          # wipe and re-import
"""
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from core.models import Buyer, Contract, Supplier

# Columns that must be read as text (otherwise pandas may turn IDs into floats like "123.0")
TEXT_COLUMNS = ["persistent_id", "tender_id", "lot_id", "country", "buyer_id", "bidder_id", "cpv2_smp", "cartel_id"]

# Numeric risk screens kept in Contract.screens (ML features).
# Deliberately NOT included: is_cartel (target), cartel_id / cartel_tender (case identifiers),
# tender_year (stored as its own column) and raw IDs.
SCREEN_COLUMNS = [
    "lot_bidscount", "lot_bidscount_avg", "lot_bidscount_avg_lag",
    "singleb_avg", "singleb_avg_lag",
    "bid_isconsortium", "bid_issubcontracted", "bid_issubcontracted_avg", "bid_issubcontracted_avg_lag",
    "relative_value", "relative_value_avg", "relative_value_avg_lag",
    "cpv3_avg_bidder_yearly", "cpv3_avg_bidder_yearly_avg_lag",
    "buyer_avg_bidder_yearly", "buyer_avg_bidder_yearly_avg_lag",
    "contract_count_bidder_yearly",
    "MAD_buyer", "benfords_buyer_yearly_avg", "benfords_buyer_yearly_avg_lag",
    "MAD_market", "benfords_market_yearly_avg", "benfords_market_yearly_avg_lag",
]

REQUIRED_COLUMNS = (
    ["persistent_id", "tender_id", "lot_id", "country", "tender_year", "buyer_id", "bidder_id",
     "is_cartel", "cpv2_smp", "cartel_id"] + SCREEN_COLUMNS
)


def clean(value):
    """NaN -> None (JSON and nullable DB columns cannot store NaN)."""
    return None if pd.isna(value) else value


class Command(BaseCommand):
    help = "Import the labelled cartel dataset (CSV) into the database."

    def add_arguments(self, parser):
        parser.add_argument("--source", required=True, help="Path to the CSV file")
        parser.add_argument("--limit", type=int, default=None, help="Import a random sample of N rows (for quick dev runs)")
        parser.add_argument("--clear", action="store_true", help="Delete existing contracts, suppliers and buyers first")
        parser.add_argument("--batch-size", type=int, default=1000)

    def handle(self, *args, **options):
        path = Path(options["source"])
        if not path.exists():
            raise CommandError(f"File not found: {path}")

        df = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, low_memory=False)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise CommandError(f"The CSV is missing expected columns: {missing_cols}")

        if options["limit"]:
            df = df.sample(n=min(options["limit"], len(df)), random_state=42)

        if Contract.objects.exists() and not options["clear"]:
            raise CommandError("Contracts already exist. Re-running would duplicate them: use --clear to start over.")

        # Rows we cannot use at all
        n_read = len(df)
        df = df.dropna(subset=["tender_id", "bidder_id", "country", "tender_year", "is_cartel", "cartel_id"])
        n_skipped = n_read - len(df)

        with transaction.atomic():
            if options["clear"]:
                # Cascades to predictions and flagged cases of those contracts
                Contract.objects.all().delete()
                Supplier.objects.all().delete()
                Buyer.objects.all().delete()

            Supplier.objects.bulk_create(
                [Supplier(bidder_id=b) for b in df["bidder_id"].unique()], ignore_conflicts=True)
            Buyer.objects.bulk_create(
                [Buyer(buyer_id=b) for b in df["buyer_id"].dropna().unique()], ignore_conflicts=True)
            supplier_pk = dict(Supplier.objects.values_list("bidder_id", "id"))
            buyer_pk = dict(Buyer.objects.values_list("buyer_id", "id"))

            contracts = []
            for row in df.to_dict("records"):
                screens = {c: clean(row[c]) for c in SCREEN_COLUMNS}
                buyer_key = clean(row["buyer_id"])
                contracts.append(Contract(
                    tender_id=row["tender_id"],
                    lot_id=clean(row["lot_id"]),
                    persistent_id=clean(row["persistent_id"]),
                    country=row["country"],
                    tender_year=int(row["tender_year"]),
                    buyer_id=buyer_pk[buyer_key] if buyer_key is not None else None,   # FK column (our pk)
                    supplier_id=supplier_pk[row["bidder_id"]],                          # FK column (our pk)
                    cpv2=clean(row["cpv2_smp"]) or "",
                    lot_bids_count=screens["lot_bidscount"],
                    relative_value=screens["relative_value"],
                    screens=screens,
                    is_cartel=bool(int(row["is_cartel"])),
                    cartel_id=row["cartel_id"],
                ))
            Contract.objects.bulk_create(contracts, batch_size=options["batch_size"])

        n_no_buyer = sum(1 for c in contracts if c.buyer_id is None)
        n_pos = sum(1 for c in contracts if c.is_cartel)
        self.stdout.write(self.style.SUCCESS(f"Imported {len(contracts)} contracts."))
        self.stdout.write(f"  rows read: {n_read} | skipped (unusable): {n_skipped}")
        self.stdout.write(f"  suppliers: {Supplier.objects.count()} | buyers: {Buyer.objects.count()}")
        if contracts:
            self.stdout.write(f"  contracts without buyer: {n_no_buyer} ({n_no_buyer / len(contracts):.1%})")
            self.stdout.write(f"  is_cartel = 1: {n_pos} ({n_pos / len(contracts):.1%})")
