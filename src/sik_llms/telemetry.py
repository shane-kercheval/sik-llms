"""OpenTelemetry utilities for sik-llms."""
import os
import warnings
from contextlib import nullcontext
from typing import Any


def is_telemetry_enabled() -> bool:
    """
    Check if OpenTelemetry is enabled via environment variables.

    Follows OpenTelemetry standard: OTEL_SDK_DISABLED=false enables telemetry.
    Default is disabled for performance and security.

    Returns:
        True if telemetry should be enabled, False otherwise.
    """
    disabled = os.getenv('OTEL_SDK_DISABLED', 'true').lower()
    return disabled in ('false', '0', 'no')


def get_tracer() -> object | None:
    """
    Get OpenTelemetry tracer if available and enabled.

    Returns:
        OpenTelemetry tracer instance or None if disabled/unavailable.
    """
    if not is_telemetry_enabled():
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        # Set up tracer provider if not already configured
        current_provider = trace.get_tracer_provider()
        if not hasattr(current_provider, '_sik_llms_initialized'):
            # Create resource with service info
            resource = Resource.create({
                'service.name': os.getenv('OTEL_SERVICE_NAME', 'sik-llms'),
                'service.version': '1.0.0',  # TODO: Get from package version
            })

            # Create and set tracer provider
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)

            # Set up OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=os.getenv(
                    'OTEL_EXPORTER_OTLP_ENDPOINT',
                    'http://localhost:4318/v1/traces',
                ),
                headers=_parse_headers(os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')),
            )

            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(span_processor)

            # Mark as initialized
            provider._sik_llms_initialized = True

        return trace.get_tracer("sik-llms")

    except ImportError:
        warnings.warn(
            "OTEL_SDK_DISABLED=false but opentelemetry not installed. "
            "Install with: pip install sik-llms[telemetry]",
            UserWarning,
            stacklevel=2,
        )
        return None


def get_meter() -> object | None:
    """
    Get OpenTelemetry meter if available and enabled.

    Returns:
        OpenTelemetry meter instance or None if disabled/unavailable.
    """
    if not is_telemetry_enabled():
        return None

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource

        # Set up meter provider if not already configured
        current_provider = metrics.get_meter_provider()
        if not hasattr(current_provider, '_sik_llms_initialized'):
            # Create resource
            resource = Resource.create({
                "service.name": os.getenv("OTEL_SERVICE_NAME", "sik-llms"),
                "service.version": "1.0.0",
            })

            # Create OTLP metric exporter
            otlp_exporter = OTLPMetricExporter(
                endpoint=os.getenv(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://localhost:4318/v1/metrics",
                ).replace("/traces", "/metrics"),
                headers=_parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")),
            )

            # Create meter provider with periodic reader
            reader = PeriodicExportingMetricReader(
                exporter=otlp_exporter,
                export_interval_millis=5000,  # Export every 5 seconds
            )

            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)

            # Mark as initialized
            provider._sik_llms_initialized = True

        return metrics.get_meter("sik-llms")

    except ImportError:
        # Silently fail for metrics if traces are working
        return None


def create_span_context(trace_id: str, span_id: str) -> object | None:
    """
    Create SpanContext from string IDs for span linking.

    Args:
        trace_id: Hexadecimal trace ID string
        span_id: Hexadecimal span ID string

    Returns:
        SpanContext object or None if invalid/unavailable
    """
    if not is_telemetry_enabled():
        return None

    try:
        from opentelemetry.trace import SpanContext

        return SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id, 16),
            is_remote=True,
        )
    except (ImportError, ValueError, TypeError):
        return None


def create_span_link(
    trace_id: str, span_id: str, attributes: dict[str, Any] | None = None,
) -> object | None:
    """
    Create span link for correlation with external spans.

    Args:
        trace_id: Original trace ID to link to
        span_id: Original span ID to link to
        attributes: Optional attributes for the link

    Returns:
        Link object or None if unavailable
    """
    if not is_telemetry_enabled():
        return None

    try:
        from opentelemetry.trace import Link

        span_context = create_span_context(trace_id, span_id)
        if span_context:
            return Link(context=span_context, attributes=attributes or {})
    except ImportError:
        pass

    return None


def _parse_headers(header_string: str) -> dict[str, str]:
    """Parse OTEL_EXPORTER_OTLP_HEADERS format."""
    if not header_string:
        return {}

    headers = {}
    for item in header_string.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers


def safe_span(tracer: object | None, name: str, **kwargs: dict) -> object:
    """
    Create span safely, returning nullcontext if tracer unavailable.

    Args:
        tracer: OpenTelemetry tracer or None
        name: Span name
        **kwargs: Additional span arguments

    Returns:
        Span context manager or nullcontext
    """
    if tracer:
        return tracer.start_as_current_span(name, **kwargs)
    return nullcontext()
