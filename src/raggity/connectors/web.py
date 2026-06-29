"""Web-crawl connector using httpx + trafilatura.

Requires the ``raggity[web]`` extra::

    pip install raggity[web]
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from . import Connector
from ..models import Document

log = logging.getLogger("raggity.connectors.web")

# Default upper bound on pages fetched in a single WebConnector.fetch() call.
_DEFAULT_MAX_PAGES = 200


# ---------------------------------------------------------------------------
# Seams — monkeypatch these in tests; lazy-import heavy deps here
# ---------------------------------------------------------------------------

def _fetch(url: str) -> str:
    """GET *url* and return the response body as a string."""
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required for web ingestion. "
            "Install it with: pip install raggity[web]"
        ) from exc
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    return resp.text


def _extract(html: str, url: str) -> tuple[str, str]:
    """Extract (title, main_text) from *html* using trafilatura."""
    try:
        import trafilatura  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "trafilatura is required for web ingestion. "
            "Install it with: pip install raggity[web]"
        ) from exc
    text = trafilatura.extract(html, url=url, include_comments=False) or ""
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else "") or urlparse(url).path or url
    return title, text


# ---------------------------------------------------------------------------
# Link extractor (stdlib only — no BeautifulSoup dependency)
# ---------------------------------------------------------------------------

class _LinkParser(HTMLParser):
    """Minimal HTMLParser that collects all href values."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Return absolute URLs for all <a href=...> links in *html*."""
    parser = _LinkParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.links:
        absolute = urljoin(base_url, href)
        # Strip fragment
        parsed = urlparse(absolute)
        clean = parsed._replace(fragment="").geturl()
        links.append(clean)
    return links


def _same_domain(url: str, origin: str) -> bool:
    return urlparse(url).netloc == urlparse(origin).netloc


# ---------------------------------------------------------------------------
# WebConnector
# ---------------------------------------------------------------------------

class WebConnector(Connector):
    """Fetch one or more web pages and return them as :class:`Document` objects.

    Parameters
    ----------
    url:
        The start URL to fetch.
    depth:
        BFS crawl depth.  ``0`` fetches only *url*; ``1`` also fetches all
        same-domain links found on *url*, etc.
    same_domain:
        When ``True`` (default) restrict BFS to URLs on the same host as
        *url*.
    max_pages:
        Maximum number of pages to fetch during BFS.  Defaults to
        :data:`_DEFAULT_MAX_PAGES` (200).  The crawl stops enqueuing new
        URLs once this many pages have been visited.  ``depth=0`` always
        fetches exactly one page regardless of ``max_pages``.
        Note: robots.txt compliance and rate limiting are not yet
        implemented — these are future work.
    """

    def __init__(self, url: str, depth: int = 0, same_domain: bool = True,
                 max_pages: int = _DEFAULT_MAX_PAGES) -> None:
        self.url = url
        self.depth = depth
        self.same_domain = same_domain
        self.max_pages = max_pages

    def fetch(self) -> list[Document]:
        docs: list[Document] = []
        visited: set[str] = set()
        # BFS queue: (url, current_depth)
        queue: deque[tuple[str, int]] = deque([(self.url, 0)])

        while queue:
            current_url, current_depth = queue.popleft()
            if current_url in visited:
                continue
            # Stop fetching new pages once the cap is reached.
            if len(visited) >= self.max_pages:
                log.warning(
                    "raggity.connectors.web: max_pages=%d reached; "
                    "stopping crawl (remaining queue size: %d).",
                    self.max_pages, len(queue) + 1,
                )
                break
            visited.add(current_url)

            html = _fetch(current_url)
            title, text = _extract(html, current_url)
            file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            docs.append(
                Document(
                    path=current_url,
                    title=title,
                    text=text,
                    file_hash=file_hash,
                    mtime=0.0,
                )
            )

            # BFS expansion
            if current_depth < self.depth:
                for link in _extract_links(html, current_url):
                    if link not in visited:
                        if not self.same_domain or _same_domain(link, self.url):
                            queue.append((link, current_depth + 1))

        return docs
