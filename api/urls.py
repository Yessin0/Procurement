from django.urls import path

from . import views

urlpatterns = [
    path("contracts/", views.ContractList.as_view(), name="contract-list"),
    path("contracts/<int:pk>/", views.ContractDetail.as_view(), name="contract-detail"),
    path("predictions/<int:contract_id>/", views.PredictionDetail.as_view(), name="prediction-detail"),
    path("clusters/", views.ClusterList.as_view(), name="cluster-list"),
    path("clusters/<int:pk>/", views.ClusterDetail.as_view(), name="cluster-detail"),
    path("cases/", views.CaseList.as_view(), name="case-list"),
    path("cases/<int:pk>/", views.CaseDetail.as_view(), name="case-detail"),
    path("metrics/summary/", views.MetricsSummary.as_view(), name="metrics-summary"),
    path("metrics/trends/", views.MetricsTrends.as_view(), name="metrics-trends"),
    path("metrics/model/", views.ModelMetrics.as_view(), name="metrics-model"),
    path("metrics/graph/", views.GraphMetrics.as_view(), name="metrics-graph"),
]