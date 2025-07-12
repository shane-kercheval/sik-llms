"""OpenTelemetry utilities for sik-llms."""
import os
from contextlib import nullcontext
from typing import Any

PACKAGE_NAME = 'sik-llms'


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

    Respects existing user configuration. Only auto-configures if no
    telemetry provider has been set up.

    Returns:
        OpenTelemetry tracer instance or None if disabled/unavailable.
    """
    if not is_telemetry_enabled():
        return None

    # Check if opentelemetry is installed before proceeding
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        raise ImportError(
            "OTEL_SDK_DISABLED=false but opentelemetry not installed. "
            "Install with: pip install sik-llms[telemetry]",
        )

    from opentelemetry import trace
    from opentelemetry.trace import NoOpTracerProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    # Check if user has already configured telemetry
    current_provider = trace.get_tracer_provider()

    if not isinstance(current_provider, NoOpTracerProvider):
        # User has already set up a real provider - respect it
        return trace.get_tracer(PACKAGE_NAME)

    # No real provider exists - set up our default configuration
    resource = Resource.create({
        'service.name': os.getenv('OTEL_SERVICE_NAME', PACKAGE_NAME),
        'service.version': _get_package_version(),
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
        # e.g. "authorization=Bearer token,x-custom-header=value"
        headers=_parse_headers(os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')),
    )

    # Add batch span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    return trace.get_tracer(PACKAGE_NAME)


def get_meter() -> object | None:
    """
    Get OpenTelemetry meter if available and enabled.

    Respects existing user configuration. Only auto-configures if no
    metrics provider has been set up.

    Returns:
        OpenTelemetry meter instance or None if disabled/unavailable.
    """
    if not is_telemetry_enabled():
        return None

    # Check if opentelemetry is installed before proceeding
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        raise ImportError(
            "OTEL_SDK_DISABLED=false but opentelemetry not installed. "
            "Install with: pip install sik-llms[telemetry]",
        )

    from opentelemetry import metrics
    from opentelemetry.metrics._internal import NoOpMeterProvider  # Note: internal API
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource

    # Check if user has already configured metrics
    current_provider = metrics.get_meter_provider()

    if not isinstance(current_provider, NoOpMeterProvider):
        # User has already set up a real provider - respect it
        return metrics.get_meter(PACKAGE_NAME)

    # No real provider exists - set up our default configuration
    resource = Resource.create({
        'service.name': os.getenv('OTEL_SERVICE_NAME', PACKAGE_NAME),
        'service.version': _get_package_version(),
    })

    # Create OTLP metric exporter
    otlp_exporter = OTLPMetricExporter(
        endpoint=os.getenv(
            'OTEL_EXPORTER_OTLP_ENDPOINT',
            'http://localhost:4318/v1/metrics',
        ).replace('/traces', '/metrics'),
        # e.g. "authorization=Bearer token,x-custom-header=value"
        headers=_parse_headers(os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')),
    )

    # Create meter provider with periodic reader
    reader = PeriodicExportingMetricReader(
        exporter=otlp_exporter,
        export_interval_millis=5000,  # Export every 5 seconds
    )

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    return metrics.get_meter(PACKAGE_NAME)


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

    # Check if opentelemetry is installed before proceeding
    try:
        from opentelemetry.trace import SpanContext
    except ImportError:
        raise ImportError(
            "OTEL_SDK_DISABLED=false but opentelemetry not installed. "
            "Install with: pip install sik-llms[telemetry]",
        )
    try:
        return SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id, 16),
            is_remote=True,
        )
    except (ValueError, TypeError):
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

    # Check if opentelemetry is installed before proceeding
    try:
        from opentelemetry.trace import Link
    except ImportError:
        raise ImportError(
            "OTEL_SDK_DISABLED=false but opentelemetry not installed. "
            "Install with: pip install sik-llms[telemetry]",
        )

    span_context = create_span_context(trace_id, span_id)
    if span_context:
        return Link(context=span_context, attributes=attributes or {})

    return None


def _get_package_version() -> str:
    """Get the package version dynamically."""
    try:
        from importlib.metadata import version
        return version(PACKAGE_NAME)
    except Exception:
        return 'unknown'


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


def extract_current_trace_context() -> tuple[str | None, str | None]:
    """
    Extract trace and span IDs from the current active span.

    Returns:
        Tuple of (trace_id, span_id) as hexadecimal strings, or (None, None) if no active span
        or telemetry is disabled.
    """
    if not is_telemetry_enabled():
        return None, None

    # Check if opentelemetry is installed before proceeding
    try:
        from opentelemetry import trace
    except ImportError:
        raise ImportError(
            "OTEL_SDK_DISABLED=false but opentelemetry not installed. "
            "Install with: pip install sik-llms[telemetry]",
        )

    try:
        # Get current span
        current_span = trace.get_current_span()
        if not current_span or not current_span.is_recording():
            return None, None

        # Get span context
        span_context = current_span.get_span_context()
        if not span_context or not span_context.is_valid:
            return None, None

        # Convert to hex strings
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')

        return trace_id, span_id

    except (AttributeError, ValueError):
        # Gracefully handle any errors
        return None, None


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
