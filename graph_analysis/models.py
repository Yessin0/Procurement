from django.db import models


class Cluster(models.Model):
    """A community of suppliers found in the supplier-buyer graph (one time window).

    Written by the offline batch command `compute_graph`, read by the API.
    `config` stores the parameters of the run (window, whether the dominant
    supplier was excluded, ...) so runs can be compared.
    """

    year_from = models.PositiveSmallIntegerField()
    year_to = models.PositiveSmallIntegerField()
    size = models.PositiveIntegerField()  # number of suppliers
    density = models.FloatField()
    n_contracts = models.PositiveIntegerField()
    n_buyers = models.PositiveIntegerField()
    mean_model_risk = models.FloatField(null=True, blank=True)
    suspicion_score = models.FloatField(null=True, blank=True)
    config = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-suspicion_score"]

    def __str__(self):
        return f"Cluster {self.pk} ({self.year_from}-{self.year_to}, {self.size} suppliers)"


class ClusterMember(models.Model):
    cluster = models.ForeignKey(Cluster, on_delete=models.CASCADE, related_name="members")
    supplier = models.ForeignKey("core.Supplier", on_delete=models.CASCADE, related_name="cluster_memberships")
    contract_count = models.PositiveIntegerField(default=0)
    n_buyers = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["cluster", "supplier"], name="uniq_member_per_cluster"),
        ]


class ClusterEdge(models.Model):
    """Link between two suppliers of a cluster. weight = number of shared buyers."""

    cluster = models.ForeignKey(Cluster, on_delete=models.CASCADE, related_name="edges")
    source = models.ForeignKey("core.Supplier", on_delete=models.CASCADE, related_name="edges_as_source")
    target = models.ForeignKey("core.Supplier", on_delete=models.CASCADE, related_name="edges_as_target")
    weight = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["cluster", "source", "target"], name="uniq_edge_per_cluster"),
        ]
