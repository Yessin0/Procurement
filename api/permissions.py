from rest_framework.permissions import SAFE_METHODS, BasePermission

ROLES = ("admin", "analyst", "viewer")


def user_role(user):
    """admin = superuser/staff or group 'admin'; analyst and viewer are groups of the same name."""
    if not user or not user.is_authenticated:
        return None
    if user.is_superuser or user.is_staff:
        return "admin"
    names = set(user.groups.values_list("name", flat=True))
    return next((role for role in ROLES if role in names), None)


class IsAnalystOrAdmin(BasePermission):
    """Contracts, predictions, clusters and cases: analysts and admins (viewers are not allowed)."""

    def has_permission(self, request, view):
        return user_role(request.user) in ("admin", "analyst")


class CanViewMetrics(BasePermission):
    """/api/metrics/*: read-only for every role."""

    def has_permission(self, request, view):
        return request.method in SAFE_METHODS and user_role(request.user) in ROLES