from rest_framework import serializers

from cases.models import FlaggedCase
from core.models import Contract
from graph_analysis.models import Cluster
from ml_pipeline.models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    contract_id = serializers.IntegerField(read_only=True)
    predicted_label = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = ["contract_id", "model_version", "risk_score", "predicted_label", "threshold", "top_features", "created_at"]

    def get_predicted_label(self, obj) -> int:
        return int(obj.predicted_label)


class ContractSerializer(serializers.ModelSerializer):
    """Ground truth (is_cartel, cartel_id) is deliberately NOT exposed."""

    buyer_id = serializers.SerializerMethodField()
    supplier_id = serializers.CharField(source="supplier.bidder_id", read_only=True)
    risk_score = serializers.FloatField(read_only=True, allow_null=True)
    case_status = serializers.CharField(read_only=True, allow_null=True)

    class Meta:
        model = Contract
        fields = ["id", "tender_id", "lot_id", "persistent_id", "country", "tender_year", "buyer_id", "supplier_id",
                  "cpv2", "lot_bids_count", "relative_value", "risk_score", "case_status"]

    def get_buyer_id(self, obj) -> str | None:
        return obj.buyer.buyer_id if obj.buyer_id else None


class ContractDetailSerializer(ContractSerializer):
    prediction = serializers.SerializerMethodField()

    class Meta(ContractSerializer.Meta):
        fields = ContractSerializer.Meta.fields + ["screens", "prediction"]

    def get_prediction(self, obj) -> dict | None:
        version = self.context.get("model_version")
        prediction = obj.predictions.filter(model_version=version).first() if version else None
        return PredictionSerializer(prediction).data if prediction else None


class ClusterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cluster
        fields = ["id", "year_from", "year_to", "size", "density", "n_contracts", "n_buyers", "buyer_concentration",
                  "mean_model_risk", "suspicion_score"]


class ClusterDetailSerializer(ClusterSerializer):
    nodes = serializers.SerializerMethodField()
    edges = serializers.SerializerMethodField()

    class Meta(ClusterSerializer.Meta):
        fields = ClusterSerializer.Meta.fields + ["nodes", "edges"]

    def get_nodes(self, obj) -> list:
        return [{"id": m.supplier.bidder_id, "label": m.supplier.bidder_id,
                 "contract_count": m.contract_count, "n_buyers": m.n_buyers}
                for m in obj.members.select_related("supplier")]

    def get_edges(self, obj) -> list:
        return [{"source": e.source.bidder_id, "target": e.target.bidder_id, "weight": int(e.weight)}   # shared buyers
                for e in obj.edges.select_related("source", "target")]


class CaseSerializer(serializers.ModelSerializer):
    contract_id = serializers.IntegerField(read_only=True, allow_null=True)
    cluster_id = serializers.IntegerField(read_only=True, allow_null=True)
    reviewed_by = serializers.SerializerMethodField()

    class Meta:
        model = FlaggedCase
        fields = ["id", "source", "contract_id", "cluster_id", "score", "status", "comment", "reviewed_by",
                  "reviewed_at", "created_at"]

    def get_reviewed_by(self, obj) -> str | None:
        return obj.reviewed_by.get_username() if obj.reviewed_by_id else None


class CaseReviewSerializer(serializers.Serializer):
    """Body of PATCH /api/cases/{id}/: only confirm or dismiss, plus an optional comment."""

    status = serializers.ChoiceField(choices=[FlaggedCase.Status.CONFIRMED, FlaggedCase.Status.DISMISSED])
    comment = serializers.CharField(required=False, allow_blank=True, max_length=2000)