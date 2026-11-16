from rest_framework.exceptions import ValidationError
from rest_framework.views import exception_handler


def api_exception_handler(exc, context):
    """Every error looks like {"detail": "human readable", "code": "machine_code"}."""
    response = exception_handler(exc, context)
    if response is None:
        return None
    data = response.data
    if isinstance(exc, ValidationError):
        response.data = {"detail": "Invalid input.", "code": "invalid_input", "errors": data}
    elif isinstance(data, dict) and "detail" in data:
        code = getattr(data["detail"], "code", None) or getattr(exc, "default_code", "error")
        response.data = {"detail": str(data["detail"]), "code": str(code)}
    return response