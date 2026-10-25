from django.contrib import admin

from .models import FlaggedCase


@admin.register(FlaggedCase)
class FlaggedCaseAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "contract", "cluster", "score", "status", "reviewed_by", "created_at")
    list_filter = ("source", "status")
    raw_id_fields = ("contract", "cluster")
