from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Prediction(models.Model):
    """Risk score of one contract for one model version.

    Written by the offline batch command `score_contracts`, read by the API.
    `top_features` holds the SHAP explanation: [{"feature", "value", "impact"}, ...]
    sorted by absolute impact (top 10).
    """

    contract = models.ForeignKey("core.Contract", on_delete=models.CASCADE, related_name="predictions")
    model_version = models.CharField(max_length=64)
    risk_score = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    predicted_label = models.BooleanField()
    threshold = models.FloatField()
    top_features = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-risk_score"]
        constraints = [
            models.UniqueConstraint(fields=["contract", "model_version"], name="uniq_prediction_per_model_version"),
        ]
        indexes = [models.Index(fields=["risk_score"])]

    def __str__(self):
        return f"Prediction {self.contract_id} / {self.model_version}: {self.risk_score:.2f}"
