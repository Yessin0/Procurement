from django.db.models import F, FloatField, OuterRef, Subquery, Value
from django.utils.dateparse import parse_date
from rest_framework.exceptions import ValidationError

from cases.models import FlaggedCase
from ml_pipeline.models import Prediction


def active_model_version(requested=None):
    """The prediction set the API shows: the one asked for, else the latest honest (-oof) set."""
    if requested:
        return requested
    base = Prediction.objects.order_by("-created_at")
    latest = base.filter(model_version__endswith="-oof").first() or base.first()
    return latest.model_version if latest else None


def annotate_contracts(queryset, version):
    """Adds risk_score (from the chosen predictions) and case_status (of the contract's case, if any)."""
    case = FlaggedCase.objects.filter(contract=OuterRef("pk"), source=FlaggedCase.Source.MODEL)
    if version is None:
        risk = Value(None, output_field=FloatField())
    else:
        risk = Subquery(Prediction.objects.filter(contract=OuterRef("pk"), model_version=version).values("risk_score")[:1])
    return queryset.annotate(risk_score=risk, case_status=Subquery(case.values("status")[:1]))


# --- query-string helpers: invalid input becomes a 400 with code "invalid_input" ---------------------------

def int_param(request, name):
    raw = request.query_params.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValidationError({name: "Must be an integer."})


def float_param(request, name):
    raw = request.query_params.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValidationError({name: "Must be a number."})


def choice_param(request, name, choices):
    raw = request.query_params.get(name)
    if raw in (None, ""):
        return None
    if raw not in choices:
        raise ValidationError({name: f"Choose one of: {', '.join(choices)}."})
    return raw


def date_param(request, name):
    raw = request.query_params.get(name)
    if raw in (None, ""):
        return None
    parsed = parse_date(raw)
    if parsed is None:
        raise ValidationError({name: "Use the format YYYY-MM-DD."})
    return parsed


def apply_ordering(request, queryset, allowed, default):
    """?ordering=-risk_score; nulls always last; id as tie-breaker so pages are stable."""
    raw = request.query_params.get("ordering") or default
    name = raw.lstrip("-")
    if name not in allowed:
        raise ValidationError({"ordering": f"Choose one of: {', '.join(sorted(allowed))} (prefix - for descending)."})
    expression = F(name).desc(nulls_last=True) if raw.startswith("-") else F(name).asc(nulls_last=True)
    return queryset.order_by(expression, "id")