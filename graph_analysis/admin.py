from django.contrib import admin

from .models import Cluster


@admin.register(Cluster)
class ClusterAdmin(admin.ModelAdmin):
    list_display = ("id", "year_from", "year_to", "size", "n_contracts", "suspicion_score")
