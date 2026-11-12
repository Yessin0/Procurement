from django.db import models


class Supplier(models.Model):
    """A bidder/supplier. `bidder_id` is the anonymised ID used in the dataset."""
    #unique bidder who submits to win contracts
    bidder_id = models.CharField(max_length=64, unique=True)

    class Meta:
        ordering = ["bidder_id"]

    def __str__(self):
        return self.bidder_id


class Buyer(models.Model):
    """A public buyer (contracting authority). `buyer_id` is the anonymised dataset ID."""
    #puts out tenders and awards public contracts 
    buyer_id = models.CharField(max_length=64, unique=True)
    
    class Meta:
        ordering = ["buyer_id"]
    #defines the human-readable representation of an object (supplier_1045)
    def __str__(self):
        return self.buyer_id


class Contract(models.Model):
    """One row of the dataset = one contract awarded to one (winning) supplier.

    The dataset has no natural key (lot_id, persistent_id and buyer_id are often
    missing), so we use an auto primary key and keep the dataset identifiers as
    plain, mostly-nullable columns.
    """

    # Identifiers from the dataset (several of them are frequently missing)
    tender_id = models.CharField(max_length=255, db_index=True)
    lot_id = models.CharField(max_length=255, null=True, blank=True)
    persistent_id = models.CharField(max_length=255 , null=True, blank=True)

    country = models.CharField(max_length=32, db_index=True)  # anonymised: country_1 ... country_7
    tender_year = models.PositiveSmallIntegerField(db_index=True) #index for matching records
    
    # NB: the FK fields below create the DB columns `buyer_id` / `supplier_id`
    # holding primary keys, not the dataset's IDs (those live on Buyer / Supplier).
    
    #one buyer can have multiple contracts 
    #one supplier can win ...
    #each contract has at most one buyer and exactly one supplier
    buyer = models.ForeignKey(Buyer, null=True, blank=True, on_delete=models.PROTECT, related_name="contracts")
    #related_name=contracts => najmou naccess reverse relationship kima supplier.contracts.all() 
    supplier = models.ForeignKey(Supplier, on_delete=models.PROTECT, related_name="contracts")
    
    cpv2 = models.CharField(max_length=16, blank=True)  # dataset column `cpv2_smp` : classifies the subject of public procurement contracts.

    #  API / filters (also present inside `screens`)
    lot_bids_count = models.FloatField(null=True, blank=True)
    relative_value = models.FloatField(null=True, blank=True)

    # All numeric risk screens of the dataset, used as ML features
    screens = models.JSONField(default=dict, blank=True)

    # EVALUATION ONLY. Never a feature, never shown in case views.
    is_cartel = models.BooleanField()
    cartel_id = models.CharField(max_length=255, db_index=True)

    class Meta:
        ordering = ["id"]
        indexes = [models.Index(fields=["country", "tender_year"])]

    def __str__(self):
        return f"Contract {self.pk} (tender {self.tender_id}, {self.country}, {self.tender_year})"
