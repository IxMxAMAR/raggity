"""Opt-in OpenTelemetry observability for raggity.

OFF by default — zero overhead, zero opentelemetry imports at base-install time.

Usage (in raggity.toml):
    [observability]
    tracing = true
    service_name = "raggity"

And set OTEL_EXPORTER_OTLP_ENDPOINT to your Phoenix / Langfuse / Jaeger endpoint.

Requires pip install raggity[otel] when tracing=true.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    # Only imported for type-checking; never at runtime unless tracing is on.
    from opentelemetry.trace import Tracer  # noqa: F401

# Module-global tracer; None = no-op (off or otel not installed).
_tracer: "Tracer | None" = None


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Generator[None, None, None]:
    """Context manager that wraps a pipeline stage in an OTel span.

    When tracing is off (``_tracer`` is None), this is a pure no-op:
    it does not import opentelemetry and adds zero overhead.

    When a tracer is active, it opens a span named *name* and sets *attrs*
    as span attributes before yielding.  The span is ended on exit regardless
    of whether an exception occurred.
    """
    if _tracer is None:
        # Fast no-op path — no opentelemetry import.
        yield
        return

    # ON path: opentelemetry must be available (installed via raggity[otel]).
    with _tracer.start_as_current_span(name) as s:
        if attrs:
            s.set_attributes(attrs)
        yield


def init_tracing(cfg: Any) -> None:  # cfg: RaggityConfig
    """Set up a TracerProvider with an OTLP exporter when tracing is enabled.

    - No-op when ``cfg.observability.tracing`` is False.
    - Idempotent: if a tracer is already active, does nothing.
    - Lazy-imports opentelemetry; raises a friendly RuntimeError if
      ``raggity[otel]`` is not installed.
    - OTLP endpoint is read from the standard
      ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable.
    """
    global _tracer

    if not cfg.observability.tracing:
        return

    if _tracer is not None:
        # Already initialised — idempotent.
        return

    try:
        from opentelemetry import trace as otel_trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OpenTelemetry packages are not installed. "
            "Enable tracing with: pip install 'raggity[otel]'"
        ) from exc

    resource = Resource.create({"service.name": cfg.observability.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter()  # endpoint from OTEL_EXPORTER_OTLP_ENDPOINT
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    _tracer = provider.get_tracer("raggity")
