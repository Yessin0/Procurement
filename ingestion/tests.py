import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from core.models import Buyer, Contract, Supplier
from ingestion.management.commands.import_data import REQUIRED_COLUMNS


def make_row(**overrides):
    row = {c: 0.5 for c in REQUIRED_COLUMNS}
    row.update(
        persistent_id=None, tender_id="t1", lot_id=None, country="country_1", tender_year=2010,
        buyer_id="b1", bidder_id="s1", is_cartel=1, cpv2_smp="45", cartel_id="country_1_1",
        bid_isconsortium=0, bid_issubcontracted=0, contract_count_bidder_yearly=3, lot_bidscount=2.0,
        relative_value=0.9,
    )
    row.update(overrides)
    return row


class ImportDataTests(TestCase):
    def write_csv(self, rows):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "sample.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def run_import(self, path, *extra):
        out = StringIO()
        call_command("import_data", "--source", path, *extra, stdout=out)
        return out.getvalue()

    def test_import_creates_rows_and_handles_missing_values(self):
        path = self.write_csv([
            make_row(tender_id="t1", bidder_id="s1", buyer_id="b1", is_cartel=1),
            make_row(tender_id="t2", bidder_id="s1", buyer_id=None, is_cartel=0, relative_value=None),
            make_row(tender_id="t3", bidder_id="s2", buyer_id="b1", is_cartel=0, lot_id="L7"),
        ])
        self.run_import(path)
        self.assertEqual(Contract.objects.count(), 3)
        self.assertEqual(Supplier.objects.count(), 2)
        self.assertEqual(Buyer.objects.count(), 1)

        no_buyer = Contract.objects.get(tender_id="t2")
        self.assertIsNone(no_buyer.buyer)
        self.assertIsNone(no_buyer.relative_value)
        self.assertIsNone(no_buyer.screens["relative_value"])
        self.assertFalse(no_buyer.is_cartel)

        self.assertEqual(Contract.objects.get(tender_id="t3").lot_id, "L7")
        self.assertIsNone(Contract.objects.get(tender_id="t1").lot_id)

    def test_labels_and_identifiers_are_not_in_screens(self):
        self.run_import(self.write_csv([make_row()]))
        screens = Contract.objects.get().screens
        for forbidden in ("is_cartel", "cartel_id", "cartel_tender", "bidder_id", "buyer_id", "tender_year"):
            self.assertNotIn(forbidden, screens)
        self.assertEqual(len(screens), 23)

    def test_rerun_refuses_without_clear_and_works_with_clear(self):
        path = self.write_csv([make_row(), make_row(tender_id="t2")])
        self.run_import(path)
        with self.assertRaises(CommandError):
            self.run_import(path)
        self.run_import(path, "--clear")
        self.assertEqual(Contract.objects.count(), 2)

    def test_limit_imports_a_sample(self):
        path = self.write_csv([make_row(tender_id=f"t{i}") for i in range(10)])
        self.run_import(path, "--limit", "4")
        self.assertEqual(Contract.objects.count(), 4)

    def test_missing_column_gives_clear_error(self):
        row = make_row()
        row.pop("singleb_avg")
        with self.assertRaises(CommandError):
            self.run_import(self.write_csv([row]))
