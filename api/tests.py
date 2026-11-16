import json
import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest import mock

from django.contrib.auth.models import Group, User
from django.core.management import call_command
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from cases.models import FlaggedCase
from core.models import Buyer, Contract, Supplier
from graph_analysis.models import Cluster, ClusterEdge, ClusterMember
from ml_pipeline.models import Prediction


def user_with_role(name, role):
    user = User.objects.create_user(name, password="pw12345!")
    if role:
        user.groups.add(Group.objects.get_or_create(name=role)[0])
    return user


def client_for(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


class ApiTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.analyst = user_with_role("alice", "analyst")
        cls.viewer = user_with_role("victor", "viewer")
        cls.nobody = user_with_role("nobody", None)

        s1, s2 = Supplier.objects.create(bidder_id="S1"), Supplier.objects.create(bidder_id="S2")
        b1 = Buyer.objects.create(buyer_id="B1")
        specs = [("country_1", 2010, s1, b1, 0.9), ("country_1", 2012, s1, None, 0.6),
                 ("country_2", 2014, s2, b1, 0.3), ("country_2", 2016, s2, None, None)]    # the last one is unscored
        cls.contracts = []
        for i, (country, year, supplier, buyer, risk) in enumerate(specs):
            contract = Contract.objects.create(
                tender_id=f"t{i}", country=country, tender_year=year, supplier=supplier, buyer=buyer,
                relative_value=0.9, screens={"singleb_avg": 0.2}, is_cartel=True, cartel_id="secret_case")
            cls.contracts.append(contract)
            if risk is not None:
                Prediction.objects.create(contract=contract, model_version="v1-oof", risk_score=risk,
                                          predicted_label=risk > 0.5, threshold=0.5,
                                          top_features=[{"feature": "singleb_avg", "value": 0.2, "impact": 0.3}])
        FlaggedCase.objects.create(source="model", contract=cls.contracts[0], score=0.9)
        FlaggedCase.objects.create(source="model", contract=cls.contracts[1], score=0.6,
                                   status="confirmed", reviewed_by=cls.analyst, reviewed_at="2020-03-05T10:00:00Z")

        cls.cluster = Cluster.objects.create(year_from=2010, year_to=2011, size=2, density=1.0, n_contracts=2, n_buyers=1,
                                             buyer_concentration=1.0, mean_model_risk=0.9, suspicion_score=0.95)
        for supplier in (s1, s2):
            ClusterMember.objects.create(cluster=cls.cluster, supplier=supplier, contract_count=1, n_buyers=1)
        ClusterEdge.objects.create(cluster=cls.cluster, source=s1, target=s2, weight=3)
        FlaggedCase.objects.create(source="cluster", cluster=cls.cluster, score=0.95)


class AccessTests(ApiTestCase):
    def test_login_is_required_and_errors_have_detail_and_code(self):
        response = APIClient().get("/api/contracts/")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(set(response.data), {"detail", "code"})
        self.assertEqual(response.data["code"], "not_authenticated")

    def test_roles(self):
        self.assertEqual(client_for(self.viewer).get("/api/metrics/summary/").status_code, 200)
        self.assertEqual(client_for(self.viewer).get("/api/contracts/").status_code, 403)
        self.assertEqual(client_for(self.viewer).get("/api/cases/").status_code, 403)
        self.assertEqual(client_for(self.nobody).get("/api/metrics/summary/").status_code, 403)
        self.assertEqual(client_for(self.analyst).get("/api/contracts/").status_code, 200)

    def test_jwt_login_flow(self):
        client = APIClient()
        token = client.post("/api/auth/token/", {"username": "alice", "password": "pw12345!"}, format="json").data["access"]
        client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
        self.assertEqual(client.get("/api/contracts/").status_code, 200)

    def test_setup_roles_command(self):
        User.objects.create_user("carol", password="x")
        call_command("setup_roles", "--user", "carol", "--role", "analyst", stdout=StringIO())
        self.assertTrue(User.objects.get(username="carol").groups.filter(name="analyst").exists())
        self.assertEqual(set(Group.objects.values_list("name", flat=True)), {"admin", "analyst", "viewer"})

    def test_openapi_schema_builds(self):
        self.assertEqual(APIClient().get("/api/schema/").status_code, 200)


class ContractTests(ApiTestCase):
    def setUp(self):
        self.client = client_for(self.analyst)

    def results(self, query=""):
        response = self.client.get(f"/api/contracts/{query}")
        self.assertEqual(response.status_code, 200, response.data)
        return response.data["results"]

    def test_shape_and_hidden_ground_truth(self):
        data = self.client.get("/api/contracts/").data
        self.assertEqual(data["count"], 4)
        self.assertEqual(set(data), {"count", "next", "previous", "results"})
        first = data["results"][0]
        self.assertEqual(set(first), {"id", "tender_id", "lot_id", "persistent_id", "country", "tender_year", "buyer_id",
                                      "supplier_id", "cpv2", "lot_bids_count", "relative_value", "risk_score", "case_status"})
        self.assertEqual((first["buyer_id"], first["supplier_id"], first["risk_score"], first["case_status"]),
                         ("B1", "S1", 0.9, "pending"))
        self.assertIsNone(self.results()[1]["buyer_id"])
        self.assertNotIn("secret_case", json.dumps(data))

    def test_filters(self):
        self.assertEqual(len(self.results("?country=country_2")), 2)
        self.assertEqual(len(self.results("?year_from=2012&year_to=2014")), 2)
        self.assertEqual(len(self.results("?min_risk=0.5")), 2)
        self.assertEqual(len(self.results("?case_status=confirmed")), 1)
        self.assertEqual(len(self.results("?supplier_id=S2")), 2)
        self.assertEqual(len(self.results("?buyer_id=B1")), 2)

    def test_ordering_puts_unscored_last(self):
        scores = [r["risk_score"] for r in self.results("?ordering=-risk_score")]
        self.assertEqual(scores, [0.9, 0.6, 0.3, None])
        scores = [r["risk_score"] for r in self.results("?ordering=risk_score")]
        self.assertEqual(scores, [0.3, 0.6, 0.9, None])

    def test_bad_input_is_a_400_with_a_code(self):
        for query in ("?min_risk=abc", "?year_from=x", "?case_status=maybe", "?ordering=cartel_id"):
            response = self.client.get(f"/api/contracts/{query}")
            self.assertEqual(response.status_code, 400, query)
            self.assertEqual(response.data["code"], "invalid_input")

    def test_pagination_with_relative_links(self):
        page1 = self.client.get("/api/contracts/?page_size=3").data
        self.assertEqual(len(page1["results"]), 3)
        self.assertTrue(page1["next"].startswith("/api/contracts/?"))
        self.assertIsNone(page1["previous"])
        self.assertEqual(self.client.get(page1["next"]).data["results"][0]["id"], self.contracts[3].id)
        self.assertEqual(self.client.get("/api/contracts/?page=9").status_code, 404)

    def test_detail_and_prediction(self):
        detail = self.client.get(f"/api/contracts/{self.contracts[0].id}/").data
        self.assertEqual(detail["screens"], {"singleb_avg": 0.2})
        self.assertEqual(detail["prediction"]["top_features"][0]["feature"], "singleb_avg")
        self.assertIsNone(self.client.get(f"/api/contracts/{self.contracts[3].id}/").data["prediction"])

        prediction = self.client.get(f"/api/predictions/{self.contracts[0].id}/").data
        self.assertEqual((prediction["risk_score"], prediction["predicted_label"], prediction["model_version"]), (0.9, 1, "v1-oof"))
        missing = self.client.get(f"/api/predictions/{self.contracts[3].id}/")
        self.assertEqual((missing.status_code, missing.data["code"]), (404, "not_found"))


class ClusterAndCaseTests(ApiTestCase):
    def setUp(self):
        self.client = client_for(self.analyst)

    def test_clusters(self):
        listing = self.client.get("/api/clusters/").data["results"][0]
        self.assertNotIn("nodes", listing)
        self.assertEqual(listing["suspicion_score"], 0.95)
        detail = self.client.get(f"/api/clusters/{self.cluster.id}/").data
        self.assertEqual({n["id"] for n in detail["nodes"]}, {"S1", "S2"})
        self.assertEqual(detail["edges"], [{"source": "S1", "target": "S2", "weight": 3}])
        self.assertEqual(self.client.get("/api/clusters/?min_score=0.99").data["count"], 0)

    def test_case_list_and_filters(self):
        self.assertEqual(self.client.get("/api/cases/").data["count"], 3)
        self.assertEqual(self.client.get("/api/cases/?status=pending").data["count"], 2)
        self.assertEqual(self.client.get("/api/cases/?source=cluster").data["count"], 1)
        self.assertEqual(self.client.get("/api/cases/?min_score=0.8").data["count"], 2)
        top = self.client.get("/api/cases/").data["results"][0]
        self.assertEqual((top["source"], top["contract_id"], top["cluster_id"]), ("cluster", None, self.cluster.id))
        self.assertEqual(self.client.get("/api/cases/?date_from=2999-01-01").data["count"], 0)

    def test_confirm_and_dismiss(self):
        case = FlaggedCase.objects.get(contract=self.contracts[0])
        response = self.client.patch(f"/api/cases/{case.id}/", {"status": "confirmed", "comment": "same buyer"}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.data["status"], response.data["comment"], response.data["reviewed_by"]),
                         ("confirmed", "same buyer", "alice"))
        self.assertIsNotNone(response.data["reviewed_at"])
        again = self.client.patch(f"/api/cases/{case.id}/", {"status": "dismissed"}, format="json")
        self.assertEqual((again.data["status"], again.data["comment"]), ("dismissed", "same buyer"))   # comment kept

    def test_review_rules(self):
        case = FlaggedCase.objects.get(contract=self.contracts[0])
        bad = self.client.patch(f"/api/cases/{case.id}/", {"status": "pending"}, format="json")
        self.assertEqual((bad.status_code, bad.data["code"]), (400, "invalid_input"))
        self.assertEqual(self.client.put(f"/api/cases/{case.id}/", {"status": "confirmed"}, format="json").status_code, 405)
        viewer = client_for(self.viewer).patch(f"/api/cases/{case.id}/", {"status": "confirmed"}, format="json")
        self.assertEqual(viewer.status_code, 403)
        case.refresh_from_db()
        self.assertEqual(case.status, "pending")


class MetricsTests(ApiTestCase):
    def setUp(self):
        self.client = client_for(self.viewer)

    def test_summary(self):
        data = self.client.get("/api/metrics/summary/").data
        self.assertEqual((data["flagged_total"], data["pending"], data["confirmed"], data["dismissed"]), (3, 2, 1, 0))
        self.assertEqual(data["confirmation_rate"], 1.0)
        self.assertEqual(data["by_country"], [{"country": "country_1", "flagged": 2}])   # contract cases only
        self.assertEqual(data["by_year"], [{"tender_year": 2010, "flagged": 1}, {"tender_year": 2012, "flagged": 1}])

    def test_trends(self):
        tender = self.client.get("/api/metrics/trends/?basis=tender").data
        self.assertEqual(tender["points"], [{"period": "2010", "flagged": 1, "confirmed": 0, "dismissed": 0},
                                            {"period": "2012", "flagged": 1, "confirmed": 1, "dismissed": 0}])
        activity = self.client.get("/api/metrics/trends/?granularity=year").data
        self.assertEqual(sum(p["flagged"] for p in activity["points"]), 3)
        self.assertIn({"period": "2020", "flagged": 0, "confirmed": 1, "dismissed": 0}, activity["points"])
        self.assertEqual(self.client.get("/api/metrics/trends/?granularity=week").status_code, 400)

    def test_model_and_graph_metrics_come_from_the_batch_commands(self):
        with tempfile.TemporaryDirectory() as tmp:     # isolated: your real metrics files are never touched
            model_path = Path(tmp) / "classifier.pkl"
            with mock.patch.dict(os.environ, {"MODEL_PATH": str(model_path)}), override_settings(BASE_DIR=Path(tmp)):
                self.assertEqual(self.client.get("/api/metrics/model/").status_code, 404)
                self.assertEqual(self.client.get("/api/metrics/graph/").status_code, 404)

                (Path(tmp) / "metrics.json").write_text(json.dumps({"model_version": "v1", "pr_auc_mean": 0.6}))
                graph_dir = Path(tmp) / "graph_analysis" / "artifacts"
                graph_dir.mkdir(parents=True)
                (graph_dir / "metrics.json").write_text(json.dumps({"nmi": 0.62}))
                self.assertEqual(self.client.get("/api/metrics/model/").data["pr_auc_mean"], 0.6)
                self.assertEqual(self.client.get("/api/metrics/graph/").data["nmi"], 0.62)