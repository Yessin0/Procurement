from urllib.parse import urlparse

from rest_framework.pagination import PageNumberPagination


class RelativePageNumberPagination(PageNumberPagination):
    """?page=1&page_size=20 -> {count, next, previous, results}; next/previous are relative links."""

    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 200

    @staticmethod
    def _relative(url):
        if url is None:
            return None
        parts = urlparse(url)
        return parts.path + (f"?{parts.query}" if parts.query else "")

    def get_next_link(self):
        return self._relative(super().get_next_link())

    def get_previous_link(self):
        return self._relative(super().get_previous_link())