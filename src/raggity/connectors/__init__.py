"""Connector framework: ABC + built-in registrations."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Document
from ..registry import register


class Connector(ABC):
    """Base class for all data-source connectors.

    Each concrete connector fetches content from a source and returns a list
    of normalised :class:`~raggity.models.Document` objects ready for chunking
    and indexing.
    """

    @abstractmethod
    def fetch(self) -> list[Document]:
        """Fetch all documents from the source and return them."""


# ---------------------------------------------------------------------------
# Built-in registrations (lazy — the dotted path is NOT imported here)
# ---------------------------------------------------------------------------
register("connector", "web", "raggity.connectors.web:WebConnector")
