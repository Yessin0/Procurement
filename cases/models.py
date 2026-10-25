from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class FlaggedCase(models.Model):
    """Something an analyst must review: a single contract (source=model) or a cluster (source=cluster)."""

    class Source(models.TextChoices):
        MODEL = "model", "Model"
        CLUSTER = "cluster", "Cluster"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CONFIRMED = "confirmed", "Confirmed"
        DISMISSED = "dismissed", "Dismissed"

    source = models.CharField(max_length=16, choices=Source.choices)
    contract = models.ForeignKey("core.Contract", null=True, blank=True, on_delete=models.CASCADE, related_name="cases")
    cluster = models.ForeignKey("graph_analysis.Cluster", null=True, blank=True, on_delete=models.CASCADE, related_name="cases")
    score = models.FloatField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    comment = models.TextField(blank=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="reviewed_cases")
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-score"]

    def clean(self):
        if self.source == self.Source.MODEL and (self.contract_id is None or self.cluster_id is not None):
            raise ValidationError("A 'model' case needs a contract and no cluster.")
        if self.source == self.Source.CLUSTER and (self.cluster_id is None or self.contract_id is not None):
            raise ValidationError("A 'cluster' case needs a cluster and no contract.")

    def __str__(self):
        return f"Case {self.pk} [{self.source}] {self.status} ({self.score:.2f})"
