from collections import defaultdict

from django.db.models import Count, Q
from django.db.models.functions import TruncMonth, TruncYear
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework.exceptions import NotFound
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from cases.models import FlaggedCase
from core.models import Contract
from graph_analysis import metrics_store
from graph_analysis.models import Cluster
from ml_pipeline import model_store
from ml_pipeline.models import Prediction

from .permissions import CanViewMetrics, IsAnalystOrAdmin
from .queries import (active_model_version, annotate_contracts, apply_ordering, choice_param, date_param,
                      float_param, int_param)
from .serializers import (CaseReviewSerializer, CaseSerializer, ClusterDetailSerializer, ClusterSerializer,
                          ContractDetailSerializer, ContractSerializer, PredictionSerializer)

STATUSES = [s.value for s in FlaggedCase.Status]
SOURCES = [s.value for s in FlaggedCase.Source]


def q(name, type_=OpenApiTypes.STR, description="", enum=None):
    return OpenApiParameter(name, type=type_, description=description, enum=enum, required=False)


# --- contracts and predictions -----------------------------------------------------------------------------

class ContractList(ListAPIView):
    serializer_class = ContractSerializer
    permission_classes = [IsAnalystOrAdmin]

    @extend_schema(parameters=[
        q("country"), q("year_from", OpenApiTypes.INT), q("year_to", OpenApiTypes.INT),
        q("buyer_id", description="Dataset buyer ID"), q("supplier_id", description="Dataset supplier ID"),
        q("min_risk", OpenApiTypes.FLOAT), q("case_status", enum=STATUSES),
        q("ordering", description="risk_score, tender_year, relative_value or id; prefix - for descending"),
        q("model_version", description="Prediction set to use (default: latest honest out-of-fold set)")])
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return Contract.objects.none()
        r = self.request
        qs = annotate_contracts(Contract.objects.select_related("buyer", "supplier"),
                                active_model_version(r.query_params.get("model_version")))
        if r.query_params.get("country"):
            qs = qs.filter(country=r.query_params["country"])
        if (year := int_param(r, "year_from")) is not None:
            qs = qs.filter(tender_year__gte=year)
        if (year := int_param(r, "year_to")) is not None:
            qs = qs.filter(tender_year__lte=year)
        if r.query_params.get("buyer_id"):
            qs = qs.filter(buyer__buyer_id=r.query_params["buyer_id"])
        if r.query_params.get("supplier_id"):
            qs = qs.filter(supplier__bidder_id=r.query_params["supplier_id"])
        if (minimum := float_param(r, "min_risk")) is not None:
            qs = qs.filter(risk_score__gte=minimum)
        if (status := choice_param(r, "case_status", STATUSES)):
            qs = qs.filter(case_status=status)
        return apply_ordering(r, qs, {"risk_score", "tender_year", "relative_value", "id"}, "id")


class ContractDetail(RetrieveAPIView):
    serializer_class = ContractDetailSerializer
    permission_classes = [IsAnalystOrAdmin]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return Contract.objects.none()
        return annotate_contracts(Contract.objects.select_related("buyer", "supplier"), self.version())

    def version(self):
        return active_model_version(self.request.query_params.get("model_version"))

    def get_serializer_context(self):
        return {**super().get_serializer_context(), "model_version": self.version()}


class PredictionDetail(RetrieveAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [IsAnalystOrAdmin]

    def get_object(self):
        version = active_model_version(self.request.query_params.get("model_version"))
        prediction = Prediction.objects.filter(contract_id=self.kwargs["contract_id"], model_version=version).first()
        if prediction is None:
            raise NotFound("No prediction for this contract.")
        return prediction


# --- clusters ----------------------------------------------------------------------------------------------

class ClusterList(ListAPIView):
    serializer_class = ClusterSerializer
    permission_classes = [IsAnalystOrAdmin]

    @extend_schema(parameters=[q("min_score", OpenApiTypes.FLOAT), q("ordering", description="suspicion_score, size, year_from or id")])
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return Cluster.objects.none()
        qs = Cluster.objects.all()
        if (minimum := float_param(self.request, "min_score")) is not None:
            qs = qs.filter(suspicion_score__gte=minimum)
        return apply_ordering(self.request, qs, {"suspicion_score", "size", "year_from", "id"}, "-suspicion_score")


class ClusterDetail(RetrieveAPIView):
    serializer_class = ClusterDetailSerializer
    permission_classes = [IsAnalystOrAdmin]
    queryset = Cluster.objects.all()


# --- cases -------------------------------------------------------------------------------------------------

class CaseList(ListAPIView):
    serializer_class = CaseSerializer
    permission_classes = [IsAnalystOrAdmin]

    @extend_schema(parameters=[
        q("status", enum=STATUSES), q("source", enum=SOURCES), q("min_score", OpenApiTypes.FLOAT),
        q("date_from", OpenApiTypes.DATE), q("date_to", OpenApiTypes.DATE),
        q("ordering", description="score, created_at or id; prefix - for descending")])
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return FlaggedCase.objects.none()
        r = self.request
        qs = FlaggedCase.objects.select_related("reviewed_by")
        if (status := choice_param(r, "status", STATUSES)):
            qs = qs.filter(status=status)
        if (source := choice_param(r, "source", SOURCES)):
            qs = qs.filter(source=source)
        if (minimum := float_param(r, "min_score")) is not None:
            qs = qs.filter(score__gte=minimum)
        if (day := date_param(r, "date_from")):
            qs = qs.filter(created_at__date__gte=day)
        if (day := date_param(r, "date_to")):
            qs = qs.filter(created_at__date__lte=day)
        return apply_ordering(r, qs, {"score", "created_at", "id"}, "-score")


class CaseDetail(RetrieveAPIView):
    serializer_class = CaseSerializer
    permission_classes = [IsAnalystOrAdmin]
    queryset = FlaggedCase.objects.select_related("reviewed_by")
    http_method_names = ["get", "patch", "head", "options"]

    @extend_schema(request=CaseReviewSerializer, responses=CaseSerializer)
    def patch(self, request, *args, **kwargs):
        case = self.get_object()
        review = CaseReviewSerializer(data=request.data)
        review.is_valid(raise_exception=True)
        case.status = review.validated_data["status"]
        if "comment" in review.validated_data:
            case.comment = review.validated_data["comment"]
        case.reviewed_by = request.user
        case.reviewed_at = timezone.now()
        case.save(update_fields=["status", "comment", "reviewed_by", "reviewed_at"])
        return Response(CaseSerializer(case).data)


# --- metrics -----------------------------------------------------------------------------------------------

class MetricsSummary(APIView):
    """Counts of cases by status. The by_country / by_year breakdowns cover contract cases only (clusters have no country)."""

    permission_classes = [CanViewMetrics]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        counts = dict(FlaggedCase.objects.values_list("status").annotate(n=Count("id")))
        pending, confirmed, dismissed = (counts.get(s, 0) for s in ("pending", "confirmed", "dismissed"))
        reviewed = confirmed + dismissed
        contract_cases = FlaggedCase.objects.filter(source=FlaggedCase.Source.MODEL)
        return Response({
            "flagged_total": pending + confirmed + dismissed,
            "pending": pending, "confirmed": confirmed, "dismissed": dismissed,
            "confirmation_rate": confirmed / reviewed if reviewed else None,
            "by_country": [{"country": r["contract__country"], "flagged": r["n"]} for r in
                           contract_cases.values("contract__country").annotate(n=Count("id")).order_by("-n", "contract__country")],
            "by_year": [{"tender_year": r["contract__tender_year"], "flagged": r["n"]} for r in
                        contract_cases.values("contract__tender_year").annotate(n=Count("id")).order_by("contract__tender_year")],
        })


class MetricsTrends(APIView):
    """basis=activity (default): cases by the month/year they were created, reviews by the date of the review.
    basis=tender: contract cases by the tender year of their contract (the data's own time axis)."""

    permission_classes = [CanViewMetrics]

    @extend_schema(parameters=[q("granularity", enum=["month", "year"]), q("basis", enum=["activity", "tender"]),
                               q("date_from", OpenApiTypes.DATE), q("date_to", OpenApiTypes.DATE)],
                   responses=OpenApiTypes.OBJECT)
    def get(self, request):
        granularity = choice_param(request, "granularity", ("month", "year")) or "month"
        basis = choice_param(request, "basis", ("activity", "tender")) or "activity"
        date_from, date_to = date_param(request, "date_from"), date_param(request, "date_to")

        if basis == "tender":
            cases = FlaggedCase.objects.filter(source=FlaggedCase.Source.MODEL)
            if date_from:
                cases = cases.filter(contract__tender_year__gte=date_from.year)
            if date_to:
                cases = cases.filter(contract__tender_year__lte=date_to.year)
            rows = cases.values("contract__tender_year").annotate(
                flagged=Count("id"), confirmed=Count("id", filter=Q(status="confirmed")),
                dismissed=Count("id", filter=Q(status="dismissed"))).order_by("contract__tender_year")
            return Response({"granularity": "year", "basis": "tender", "points": [
                {"period": str(r["contract__tender_year"]), "flagged": r["flagged"],
                 "confirmed": r["confirmed"], "dismissed": r["dismissed"]} for r in rows]})

        trunc, fmt = (TruncMonth, "%Y-%m") if granularity == "month" else (TruncYear, "%Y")
        created, reviewed = FlaggedCase.objects.all(), FlaggedCase.objects.exclude(reviewed_at=None)
        if date_from:
            created, reviewed = created.filter(created_at__date__gte=date_from), reviewed.filter(reviewed_at__date__gte=date_from)
        if date_to:
            created, reviewed = created.filter(created_at__date__lte=date_to), reviewed.filter(reviewed_at__date__lte=date_to)

        points = defaultdict(lambda: {"flagged": 0, "confirmed": 0, "dismissed": 0})
        for r in created.annotate(p=trunc("created_at")).values("p").annotate(n=Count("id")):
            points[r["p"].strftime(fmt)]["flagged"] += r["n"]
        for r in reviewed.annotate(p=trunc("reviewed_at")).values("p", "status").annotate(n=Count("id")):
            if r["status"] in ("confirmed", "dismissed"):
                points[r["p"].strftime(fmt)][r["status"]] += r["n"]
        return Response({"granularity": granularity, "basis": "activity",
                         "points": [{"period": p, **points[p]} for p in sorted(points)]})


class ModelMetrics(APIView):
    permission_classes = [CanViewMetrics]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        try:
            return Response(model_store.load_metrics())
        except FileNotFoundError as exc:
            raise NotFound(str(exc))


class GraphMetrics(APIView):
    permission_classes = [CanViewMetrics]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        try:
            return Response(metrics_store.load_metrics())
        except FileNotFoundError as exc:
            raise NotFound(str(exc))