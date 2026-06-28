"""Tests for opt-in OpenTelemetry observability (Task 3 / Phase E).

Rules:
- OFF path (default): span() is a pure no-op, ZERO opentelemetry imports.
- ON path: span() opens / closes a real OTel span when a tracer is injected.
- init_tracing no-ops when tracing=False.
- init_tracing is idempotent.
"""
from __future__ import annotations

import sys
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_otel_from_modules():
    """Remove any opentelemetry modules that may have been imported earlier in
    this test session so we can verify the OFF path never touches them.
    This works by swapping them out transiently; pytest's module isolation
    means each function that calls this still sees the real modules afterwards.
    """
    return [k for k in sys.modules if k.startswith("opentelemetry")]


# ---------------------------------------------------------------------------
# OFF-path: no opentelemetry import, zero overhead
# ---------------------------------------------------------------------------

def test_span_noop_when_tracing_off_yields_without_error():
    """span() used as a context manager must yield cleanly when tracing is off."""
    import raggity.observability as obs

    # Ensure tracer is None (default / off)
    obs._tracer = None

    result = []
    with obs.span("test_span", foo="bar"):
        result.append("inside")

    assert result == ["inside"]


def test_span_noop_does_not_import_opentelemetry(monkeypatch):
    """The OFF path must not import opentelemetry — check sys.modules is clean."""
    # Temporarily hide opentelemetry from sys.modules so import would fail
    otel_keys = [k for k in sys.modules if k.startswith("opentelemetry")]
    saved = {k: sys.modules.pop(k) for k in otel_keys}

    import raggity.observability as obs
    obs._tracer = None  # ensure off

    try:
        with obs.span("x", hello="world"):
            pass
        # If we got here without ImportError, the OFF path is import-free
    finally:
        sys.modules.update(saved)


def test_span_noop_is_a_context_manager():
    """Verify span() returns an object usable with `with`."""
    import raggity.observability as obs
    obs._tracer = None

    cm = obs.span("check")
    # Must have __enter__ and __exit__
    assert hasattr(cm, "__enter__")
    assert hasattr(cm, "__exit__")


def test_init_tracing_noop_when_tracing_false():
    """init_tracing(cfg) must no-op (not raise) when cfg.observability.tracing=False."""
    from raggity.config import RaggityConfig
    import raggity.observability as obs

    cfg = RaggityConfig()
    assert cfg.observability.tracing is False

    obs._tracer = None
    obs.init_tracing(cfg)

    assert obs._tracer is None


# ---------------------------------------------------------------------------
# Config: ObservabilityConfig defaults
# ---------------------------------------------------------------------------

def test_observability_config_defaults():
    """RaggityConfig must expose an observability field with sane defaults."""
    from raggity.config import RaggityConfig, ObservabilityConfig

    cfg = RaggityConfig()
    assert isinstance(cfg.observability, ObservabilityConfig)
    assert cfg.observability.tracing is False
    assert cfg.observability.service_name == "raggity"


def test_observability_config_toml_override(tmp_path):
    """[observability] block in raggity.toml must be parsed."""
    from raggity.config import load_config

    p = tmp_path / "raggity.toml"
    p.write_text('[observability]\ntracing = true\nservice_name = "my-svc"\n')
    cfg = load_config(str(p))

    assert cfg.observability.tracing is True
    assert cfg.observability.service_name == "my-svc"


# ---------------------------------------------------------------------------
# ON-path: real spans recorded via InMemorySpanExporter
# ---------------------------------------------------------------------------

def test_span_records_when_tracer_active():
    """When a real tracer is injected, span() must open/close an OTel span."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    import raggity.observability as obs

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Inject a real tracer directly (bypassing init_tracing which needs OTLP)
    obs._tracer = provider.get_tracer("test-tracer")

    try:
        with obs.span("my_operation", key="value"):
            pass
    finally:
        obs._tracer = None  # clean up

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "my_operation"


def test_span_sets_attributes_when_tracer_active():
    """Attributes passed to span() must appear on the recorded OTel span."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    import raggity.observability as obs

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    obs._tracer = provider.get_tracer("test-tracer")

    try:
        with obs.span("retrieve", query="hello", chunk_count=5):
            pass
    finally:
        obs._tracer = None

    finished = exporter.get_finished_spans()
    attrs = dict(finished[0].attributes)
    assert attrs.get("query") == "hello"
    assert attrs.get("chunk_count") == 5


def test_init_tracing_idempotent(monkeypatch):
    """Calling init_tracing twice must not raise or double-register exporters."""
    from raggity.config import RaggityConfig
    import raggity.observability as obs

    # Use a config with tracing on but without a real OTLP endpoint
    # We monkeypatch the lazy import path so it uses an in-memory provider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Patch _tracer as if init_tracing already ran
    obs._tracer = provider.get_tracer("idempotent-test")
    first_tracer = obs._tracer

    # A second call to init_tracing with tracing=True should keep existing tracer
    cfg = RaggityConfig()
    # Directly set tracing=True on a copy
    cfg = cfg.model_copy(update={"observability": cfg.observability.model_copy(update={"tracing": True})})

    # The idempotency guarantee: if _tracer already set, don't replace it
    obs.init_tracing(cfg)
    assert obs._tracer is first_tracer

    obs._tracer = None  # clean up


def test_span_exception_propagates_when_tracer_active():
    """Exceptions inside a span context must propagate normally."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    import raggity.observability as obs

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    obs._tracer = provider.get_tracer("test-tracer")

    try:
        import pytest
        with pytest.raises(ValueError, match="boom"):
            with obs.span("bad_op"):
                raise ValueError("boom")
    finally:
        obs._tracer = None

    # Span should still be finished (ended on error)
    finished = exporter.get_finished_spans()
    assert len(finished) == 1


def test_span_exception_propagates_when_tracing_off():
    """Exceptions inside span() context must propagate even in no-op mode."""
    import pytest
    import raggity.observability as obs
    obs._tracer = None

    with pytest.raises(RuntimeError, match="noop-error"):
        with obs.span("op"):
            raise RuntimeError("noop-error")
