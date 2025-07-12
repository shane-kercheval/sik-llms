"""Tests for OpenTelemetry integration."""
import os
import pytest
from unittest.mock import patch, MagicMock
from sik_llms import create_client
from sik_llms.telemetry import is_telemetry_enabled, get_tracer, get_meter


class TestTelemetryConfiguration:
    """Test telemetry configuration and setup."""

    def test_telemetry_disabled_by_default(self):
        """Ensure telemetry is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            assert not is_telemetry_enabled()

    def test_telemetry_enabled_with_env_var(self):
        """Test telemetry enabled via environment variable."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            assert is_telemetry_enabled()

    def test_telemetry_disabled_explicitly(self):
        """Test telemetry explicitly disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            assert not is_telemetry_enabled()

    def test_telemetry_enabled_with_zero(self):
        """Test telemetry enabled with '0'."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "0"}):
            assert is_telemetry_enabled()

    def test_telemetry_enabled_with_no(self):
        """Test telemetry enabled with 'no'."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "no"}):
            assert is_telemetry_enabled()


class TestTelemetryIntegration:
    """Test telemetry integration with sik-llms."""

    def test_client_creation_without_telemetry(self):
        """Ensure client works normally without telemetry."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = create_client("gpt-4o-mini")
            assert client.tracer is None
            assert client.meter is None

    @patch('sik_llms.models_base.get_tracer')
    @patch('sik_llms.models_base.get_meter')
    def test_client_creation_with_telemetry(self, mock_meter, mock_tracer):  # noqa: ANN001
        """Test client creation with telemetry enabled."""
        mock_tracer_instance = MagicMock()
        mock_meter_instance = MagicMock()
        mock_tracer.return_value = mock_tracer_instance
        mock_meter.return_value = mock_meter_instance

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            client = create_client("gpt-4o-mini")
            assert client.tracer is mock_tracer_instance
            assert client.meter is mock_meter_instance

    def test_graceful_degradation_missing_otel(self):
        """Test that functions raise ImportError when OpenTelemetry not installed but enabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(
                    ImportError, match="OTEL_SDK_DISABLED=false but opentelemetry not installed",
                ):
                    get_tracer()
                with pytest.raises(
                    ImportError, match="OTEL_SDK_DISABLED=false but opentelemetry not installed",
                ):
                    get_meter()

    def test_get_tracer_returns_none_when_disabled(self):
        """Test get_tracer returns None when telemetry disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            tracer = get_tracer()
            assert tracer is None

    def test_get_meter_returns_none_when_disabled(self):
        """Test get_meter returns None when telemetry disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            meter = get_meter()
            assert meter is None


class TestTokenSummaryMetrics:
    """Test TokenSummary metrics emission."""

    @patch('sik_llms.telemetry.get_meter')
    def test_metrics_emission(self, mock_get_meter):  # noqa: ANN001
        """Test that TokenSummary emits metrics correctly."""
        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_histogram = MagicMock()

        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_get_meter.return_value = mock_meter

        from sik_llms.models_base import TokenSummary

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
            input_cost=0.001,
            output_cost=0.002,
        )

        summary.emit_metrics(mock_meter, {"model": "gpt-4"})

        # Verify metrics were created and recorded
        assert mock_meter.create_counter.call_count >= 3  # input, output, cost
        assert mock_meter.create_histogram.called  # duration
        assert mock_counter.add.call_count >= 3
        assert mock_histogram.record.called

    def test_metrics_emission_with_cache_tokens(self):
        """Test metrics emission with cache tokens."""
        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_histogram = MagicMock()

        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram

        from sik_llms.models_base import TokenSummary

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
            cache_read_tokens=25,
            cache_write_tokens=10,
        )

        summary.emit_metrics(mock_meter, {"model": "claude-3"})

        # Should have created cache metrics too
        assert mock_meter.create_counter.call_count >= 4  # input, output, cache_read, cache_write
        assert mock_counter.add.call_count >= 4

    def test_metrics_emission_no_meter(self):
        """Test metrics emission gracefully handles no meter."""
        from sik_llms.models_base import TokenSummary

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
        )

        # Should not raise exception
        summary.emit_metrics(None, {"model": "gpt-4"})

    def test_metrics_emission_handles_exceptions(self):
        """Test metrics emission handles exceptions gracefully."""
        mock_meter = MagicMock()
        mock_meter.create_counter.side_effect = Exception("Test exception")

        from sik_llms.models_base import TokenSummary

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
        )

        # Should not raise exception
        summary.emit_metrics(mock_meter, {"model": "gpt-4"})


class TestSafeSpan:
    """Test safe_span utility function."""

    def test_safe_span_with_tracer(self):
        """Test safe_span with valid tracer."""
        from sik_llms.telemetry import safe_span

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        result = safe_span(mock_tracer, "test_span", attributes={"key": "value"})

        mock_tracer.start_as_current_span.assert_called_once_with(
            "test_span", attributes={"key": "value"},
        )
        assert result == mock_span

    def test_safe_span_without_tracer(self):
        """Test safe_span with None tracer."""
        from sik_llms.telemetry import safe_span
        from contextlib import nullcontext

        result = safe_span(None, "test_span", attributes={"key": "value"})

        # Should return nullcontext
        assert isinstance(result, type(nullcontext()))


class TestSpanLinking:
    """Test span linking functionality."""

    def test_create_span_context_disabled(self):
        """Test create_span_context when telemetry disabled."""
        from sik_llms.telemetry import create_span_context

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            result = create_span_context("1234567890abcdef", "fedcba0987654321")
            assert result is None

    def test_create_span_link_disabled(self):
        """Test create_span_link when telemetry disabled."""
        from sik_llms.telemetry import create_span_link

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            result = create_span_link("1234567890abcdef", "fedcba0987654321")
            assert result is None

    @patch('sik_llms.telemetry.is_telemetry_enabled')
    def test_create_span_context_import_error(self, mock_enabled):   # noqa
        """Test create_span_context raises ImportError when dependencies missing."""
        from sik_llms.telemetry import create_span_context

        mock_enabled.return_value = True

        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(
                ImportError, match="OTEL_SDK_DISABLED=false but opentelemetry not installed",
            ):
                create_span_context("1234567890abcdef", "fedcba0987654321")

    @patch('sik_llms.telemetry.is_telemetry_enabled')
    def test_create_span_link_import_error(self, mock_enabled):   # noqa
        """Test create_span_link raises ImportError when dependencies missing."""
        from sik_llms.telemetry import create_span_link

        mock_enabled.return_value = True

        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(
                ImportError, match="OTEL_SDK_DISABLED=false but opentelemetry not installed",
            ):
                create_span_link("1234567890abcdef", "fedcba0987654321")


class TestStandardProviderDetection:
    """Test that we properly detect and respect existing OpenTelemetry setup."""

    @patch('sik_llms.telemetry.is_telemetry_enabled')
    def test_respects_existing_tracer_provider(self, mock_enabled):   # noqa
        """Test that we don't override existing user TracerProvider."""
        mock_enabled.return_value = True

        # User sets up their own provider
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        user_provider = TracerProvider()
        trace.set_tracer_provider(user_provider)

        # Our code should respect their provider
        tracer = get_tracer()

        # Verify we got a tracer from their provider, not a new one
        assert trace.get_tracer_provider() is user_provider
        assert tracer is not None

    @patch('sik_llms.telemetry.is_telemetry_enabled')
    def test_auto_configures_when_no_provider(self, mock_enabled):   # noqa
        """Test that we auto-configure when only NoOpTracerProvider exists."""
        mock_enabled.return_value = True

        # Reset to default NoOp state
        from opentelemetry.trace import NoOpTracerProvider

        # Simulate fresh start - no user configuration
        with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
            mock_get_provider.return_value = NoOpTracerProvider()

            with patch('opentelemetry.trace.set_tracer_provider') as mock_set_provider:
                tracer = get_tracer()

                # Verify we attempted to set up our own provider
                assert mock_set_provider.called
                assert tracer is not None

    @patch('sik_llms.telemetry.is_telemetry_enabled')
    def test_no_interference_with_user_config(self, mock_enabled):   # noqa
        """Test that we don't interfere with user's custom configuration."""
        mock_enabled.return_value = True

        # Mock the OpenTelemetry provider to simulate user's configuration
        with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
            from opentelemetry.sdk.trace import TracerProvider

            # User's provider is not a NoOpTracerProvider
            user_provider = MagicMock(spec=TracerProvider)
            mock_get_provider.return_value = user_provider

            # Mock trace.get_tracer to return a tracer
            with patch('opentelemetry.trace.get_tracer') as mock_get_tracer:
                mock_tracer = MagicMock()
                mock_get_tracer.return_value = mock_tracer

                # Our code runs
                tracer = get_tracer()

                # Verify we didn't try to set a new provider
                from opentelemetry import trace
                with patch.object(trace, 'set_tracer_provider') as mock_set:
                    # Re-run to verify no set_tracer_provider call
                    get_tracer()
                    mock_set.assert_not_called()

                # Verify we returned a tracer from the existing provider
                assert tracer is mock_tracer

    def test_handles_missing_opentelemetry_gracefully(self):
        """Test that get_tracer raises ImportError when OpenTelemetry not installed but enabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(
                    ImportError, match="OTEL_SDK_DISABLED=false but opentelemetry not installed",
                ):
                    get_tracer()


class TestHeaderParsing:
    """Test OTLP header parsing."""

    def test_parse_headers_empty(self):
        """Test parsing empty header string."""
        from sik_llms.telemetry import _parse_headers

        result = _parse_headers("")
        assert result == {}

    def test_parse_headers_single(self):
        """Test parsing single header."""
        from sik_llms.telemetry import _parse_headers

        result = _parse_headers("authorization=Bearer token123")
        assert result == {"authorization": "Bearer token123"}

    def test_parse_headers_multiple(self):
        """Test parsing multiple headers."""
        from sik_llms.telemetry import _parse_headers

        result = _parse_headers("authorization=Bearer token123,x-api-key=key456")
        expected = {
            "authorization": "Bearer token123",
            "x-api-key": "key456",
        }
        assert result == expected

    def test_parse_headers_with_spaces(self):
        """Test parsing headers with spaces."""
        from sik_llms.telemetry import _parse_headers

        result = _parse_headers(" authorization = Bearer token123 , x-api-key = key456 ")
        expected = {
            "authorization": "Bearer token123",
            "x-api-key": "key456",
        }
        assert result == expected

    def test_parse_headers_malformed(self):
        """Test parsing malformed headers."""
        from sik_llms.telemetry import _parse_headers

        result = _parse_headers("no_equals_sign,authorization=Bearer token123")
        expected = {"authorization": "Bearer token123"}
        assert result == expected


def _otel_available() -> bool:
    """Check if OpenTelemetry is available for testing."""
    try:
        import opentelemetry  # noqa
        return True
    except ImportError:
        return False


def _jaeger_available() -> bool:
    """Check if Jaeger is running locally."""
    try:
        import requests
        requests.get("http://localhost:16686/api/services", timeout=2)
        return True
    except:  # noqa: E722
        return False


@pytest.mark.integration
class TestTelemetryIntegrationReal:
    """Integration tests with real OpenTelemetry components."""

    @pytest.mark.skipif(
        not _otel_available(),
        reason="OpenTelemetry not installed",
    )
    def test_tracer_setup_with_real_otel(self):
        """Test tracer setup with real OpenTelemetry."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            # Clear any existing tracer provider
            with patch('opentelemetry.trace.get_tracer_provider') as mock_provider:
                from opentelemetry.trace import NoOpTracerProvider
                mock_provider.return_value = NoOpTracerProvider()

                tracer = get_tracer()
                # Should not be None when OpenTelemetry is available
                assert tracer is not None

    @pytest.mark.skipif(
        not _otel_available(),
        reason="OpenTelemetry not installed",
    )
    def test_meter_setup_with_real_otel(self):
        """Test meter setup with real OpenTelemetry."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            # Clear any existing meter provider
            with patch('opentelemetry.metrics.get_meter_provider') as mock_provider:
                from opentelemetry.metrics._internal import NoOpMeterProvider
                mock_provider.return_value = NoOpMeterProvider()

                meter = get_meter()
                # Should not be None when OpenTelemetry is available
                assert meter is not None


@pytest.mark.integration
class TestUserConfigurationRespect:
    """Integration tests for respecting user OpenTelemetry configuration."""

    @pytest.mark.skipif(not _otel_available(), reason="OpenTelemetry not installed")
    def test_end_to_end_user_config_preserved(self):
        """Test that user's end-to-end configuration is preserved."""
        # Mock scenario where user has already set up OpenTelemetry
        with patch('opentelemetry.trace.get_tracer_provider') as mock_get_provider:
            from opentelemetry.sdk.trace import TracerProvider

            # Simulate that user has a real provider configured
            user_provider = MagicMock(spec=TracerProvider)
            mock_get_provider.return_value = user_provider

            with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
                # Verify sik-llms respects existing configuration
                with patch('opentelemetry.trace.set_tracer_provider') as mock_set:
                    client = create_client("gpt-4o-mini")

                    # Should not have tried to set a new provider
                    mock_set.assert_not_called()

                    # Should have gotten a tracer
                    assert client.tracer is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not _jaeger_available(),
    reason="Jaeger not running - start with: make jaeger-start",
)
class TestTelemetryJaegerIntegration:
    """Integration tests requiring Jaeger instance."""

    def test_jaeger_connectivity(self):
        """Test basic Jaeger connectivity."""
        import requests

        response = requests.get("http://localhost:16686/api/services", timeout=5)
        assert response.status_code == 200

        # Should be able to query traces
        response = requests.get("http://localhost:16686/api/traces?service=test", timeout=5)
        assert response.status_code == 200
