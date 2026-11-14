"""Where the graph validation metrics live (written by compute_graph, read by GET /api/metrics/graph/)."""
import json
from pathlib import Path

from django.conf import settings


def metrics_path() -> Path:
    return Path(settings.BASE_DIR) / "graph_analysis" / "artifacts" / "metrics.json"


def save(metrics: dict) -> None:
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def load_metrics() -> dict:
    path = metrics_path()
    if not path.exists():
        raise FileNotFoundError(f"No graph metrics at {path}. Run: python manage.py compute_graph")
    return json.loads(path.read_text())