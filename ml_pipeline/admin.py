from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("contract", "model_version", "risk_score", "predicted_label", "created_at")
    list_filter = ("model_version", "predicted_label")
    raw_id_fields = ("contract",)
