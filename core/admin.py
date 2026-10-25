from django.contrib import admin
from django.db.models import Count

from .models import Buyer, Contract, Supplier


@admin.register(Supplier)
class SupplierAdmin(admin.ModelAdmin):
    list_display = ("bidder_id", "contract_count")
    search_fields = ("bidder_id",)

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(n_contracts=Count("contracts"))

    @admin.display(ordering="n_contracts", description="Contracts")
    def contract_count(self, obj):
        return obj.n_contracts


@admin.register(Buyer)
class BuyerAdmin(admin.ModelAdmin):
    list_display = ("buyer_id", "contract_count")
    search_fields = ("buyer_id",)

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(n_contracts=Count("contracts"))

    @admin.display(ordering="n_contracts", description="Contracts")
    def contract_count(self, obj):
        return obj.n_contracts


@admin.register(Contract)
class ContractAdmin(admin.ModelAdmin):
    list_display = ("id", "tender_id", "country", "tender_year", "supplier", "buyer", "cpv2", "relative_value", "is_cartel")
    list_filter = ("country", "tender_year", "cpv2", "is_cartel")
    search_fields = ("tender_id", "supplier__bidder_id", "buyer__buyer_id", "cartel_id")
    list_select_related = ("supplier", "buyer")
    raw_id_fields = ("supplier", "buyer")
    list_per_page = 50
