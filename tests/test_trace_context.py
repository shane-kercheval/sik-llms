"""Tests for TraceContext functionality."""
import os
from unittest.mock import Mock, patch
from sik_llms.models_base import (
    TraceContext,
    TokenSummary,
    TextResponse,
    ToolPredictionResponse,
    StructuredOutputResponse,
)
from sik_llms.telemetry import extract_current_trace_context


class TestTraceContext:
    """Tests for the TraceContext model."""

    def test_trace_context_creation(self):
        """Test creating a TraceContext with trace_id and span_id."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )
        assert trace_context.trace_id == '0123456789abcdef0123456789abcdef'
        assert trace_context.span_id == '0123456789abcdef'

    def test_trace_context_optional_fields(self):
        """Test that TraceContext fields are optional."""
        trace_context = TraceContext()
        assert trace_context.trace_id is None
        assert trace_context.span_id is None

    def test_trace_context_create_link_with_valid_context(self):
        """Test creating a span link from valid trace context."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )

        with patch('sik_llms.models_base.create_span_link') as mock_create_link:
            mock_link = Mock()
            mock_create_link.return_value = mock_link

            result = trace_context.create_link({'test': 'attribute'})

            mock_create_link.assert_called_once_with(
                '0123456789abcdef0123456789abcdef',
                '0123456789abcdef',
                {'test': 'attribute'},
            )
            assert result == mock_link

    def test_trace_context_create_link_with_incomplete_context(self):
        """Test that create_link returns None if trace_id or span_id is missing."""
        # Missing both
        trace_context = TraceContext()
        assert trace_context.create_link() is None

        # Missing span_id
        trace_context = TraceContext(trace_id='0123456789abcdef0123456789abcdef')
        assert trace_context.create_link() is None

        # Missing trace_id
        trace_context = TraceContext(span_id='0123456789abcdef')
        assert trace_context.create_link() is None

    def test_trace_context_serialization(self):
        """Test that TraceContext can be serialized and deserialized."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )

        # Test model_dump
        data = trace_context.model_dump()
        assert data['trace_id'] == '0123456789abcdef0123456789abcdef'
        assert data['span_id'] == '0123456789abcdef'

        # Test model_validate
        restored = TraceContext.model_validate(data)
        assert restored.trace_id == trace_context.trace_id
        assert restored.span_id == trace_context.span_id


class TestResponseObjectsWithTraceContext:
    """Tests for response objects with trace context."""

    def test_text_response_with_trace_context(self):
        """Test TextResponse can include trace context."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )

        response = TextResponse(
            response="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
            trace_context=trace_context,
        )

        assert response.trace_context == trace_context
        assert response.trace_context.trace_id == '0123456789abcdef0123456789abcdef'

    def test_tool_prediction_response_with_trace_context(self):
        """Test ToolPredictionResponse can include trace context."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )

        response = ToolPredictionResponse(
            tool_prediction=None,
            message="No tool called",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
            trace_context=trace_context,
        )

        assert response.trace_context == trace_context

    def test_structured_output_response_with_trace_context(self):
        """Test StructuredOutputResponse can include trace context."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )

        response = StructuredOutputResponse(
            parsed=None,
            refusal="Refused to parse",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
            trace_context=trace_context,
        )

        assert response.trace_context == trace_context

    def test_response_serialization_with_trace_context(self):
        """Test that responses with trace context can be serialized."""
        trace_context = TraceContext(
            trace_id='0123456789abcdef0123456789abcdef',
            span_id='0123456789abcdef',
        )

        response = TextResponse(
            response="Test",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
            trace_context=trace_context,
        )

        # Test model_dump
        data = response.model_dump()
        assert 'trace_context' in data
        assert data['trace_context']['trace_id'] == '0123456789abcdef0123456789abcdef'
        assert data['trace_context']['span_id'] == '0123456789abcdef'

        # Test JSON serialization
        json_str = response.model_dump_json()
        assert '0123456789abcdef0123456789abcdef' in json_str
        assert '0123456789abcdef' in json_str

    def test_response_without_trace_context(self):
        """Test that responses work without trace context (backward compatibility)."""
        response = TextResponse(
            response="Hello",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
        )

        assert response.trace_context is None

        # Should still serialize correctly
        data = response.model_dump()
        assert 'trace_context' in data
        assert data['trace_context'] is None


class TestExtractCurrentTraceContext:
    """Tests for extract_current_trace_context function."""

    @patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'true'})
    def test_extract_when_telemetry_disabled(self):
        """Test that extraction returns None when telemetry is disabled."""
        trace_id, span_id = extract_current_trace_context()
        assert trace_id is None
        assert span_id is None

    @patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'false'})
    def test_extract_with_valid_span(self):
        """Test extraction with a valid active span."""
        # Mock span with valid context
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span_context = Mock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0x0123456789abcdef0123456789abcdef
        mock_span_context.span_id = 0x0123456789abcdef
        mock_span.get_span_context.return_value = mock_span_context

        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            trace_id, span_id = extract_current_trace_context()

        assert trace_id == '0123456789abcdef0123456789abcdef'
        assert span_id == '0123456789abcdef'

    @patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'false'})
    def test_extract_with_no_active_span(self):
        """Test extraction when no span is active."""
        with patch('opentelemetry.trace.get_current_span', return_value=None):
            trace_id, span_id = extract_current_trace_context()

        assert trace_id is None
        assert span_id is None

    @patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'false'})
    def test_extract_with_non_recording_span(self):
        """Test extraction with a non-recording span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = False

        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            trace_id, span_id = extract_current_trace_context()

        assert trace_id is None
        assert span_id is None

    @patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'false'})
    def test_extract_with_invalid_span_context(self):
        """Test extraction with invalid span context."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span_context = Mock()
        mock_span_context.is_valid = False
        mock_span.get_span_context.return_value = mock_span_context

        with patch('opentelemetry.trace.get_current_span', return_value=mock_span):
            trace_id, span_id = extract_current_trace_context()

        assert trace_id is None
        assert span_id is None

    @patch.dict(os.environ, {'OTEL_SDK_DISABLED': 'false'})
    def test_extract_with_import_error(self):
        """Test extraction when OpenTelemetry is not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            trace_id, span_id = extract_current_trace_context()

            assert trace_id is None
            assert span_id is None


class TestClientTraceContextIntegration:
    """Tests for Client class trace context integration."""

    def test_add_trace_context_helper_method(self):
        """Test the _add_trace_context helper method."""
        from sik_llms.models_base import Client

        # Create a mock client
        client = Mock(spec=Client)
        client._add_trace_context = Client._add_trace_context.__get__(client, Client)

        # Mock response
        response = TextResponse(
            response="Test",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
        )

        # Mock extract_current_trace_context
        with patch('sik_llms.models_base.extract_current_trace_context') as mock_extract:
            mock_extract.return_value = ('0123456789abcdef0123456789abcdef', '0123456789abcdef')

            client._add_trace_context(response)

            assert response.trace_context is not None
            assert response.trace_context.trace_id == '0123456789abcdef0123456789abcdef'
            assert response.trace_context.span_id == '0123456789abcdef'

    def test_add_trace_context_with_no_context(self):
        """Test _add_trace_context when no trace context is available."""
        from sik_llms.models_base import Client

        client = Mock(spec=Client)
        client._add_trace_context = Client._add_trace_context.__get__(client, Client)

        response = TextResponse(
            response="Test",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=0.5,
        )

        with patch('sik_llms.models_base.extract_current_trace_context') as mock_extract:
            mock_extract.return_value = (None, None)

            client._add_trace_context(response)

            assert response.trace_context is None

    def test_add_trace_context_with_object_without_trace_context(self):
        """Test _add_trace_context with object that doesn't have trace_context."""
        from sik_llms.models_base import Client

        client = Mock(spec=Client)
        client._add_trace_context = Client._add_trace_context.__get__(client, Client)

        # Mock object without trace_context attribute
        response = Mock()
        delattr(response, 'trace_context')

        with patch('sik_llms.models_base.extract_current_trace_context') as mock_extract:
            mock_extract.return_value = ('0123456789abcdef0123456789abcdef', '0123456789abcdef')

            # Should not raise exception
            client._add_trace_context(response)

            # Should not have added trace_context
            assert not hasattr(response, 'trace_context')


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_existing_code_still_works(self):
        """Test that existing code without trace context continues to work."""
        # Create responses the old way (without trace_context)
        text_response = TextResponse(
            response="Hello",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=1.0,
        )

        tool_response = ToolPredictionResponse(
            tool_prediction=None,
            message="No tool",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=1.0,
        )

        structured_response = StructuredOutputResponse(
            parsed=None,
            refusal="Refused",
            input_tokens=10,
            output_tokens=5,
            duration_seconds=1.0,
        )

        # All should have trace_context as None by default
        assert text_response.trace_context is None
        assert tool_response.trace_context is None
        assert structured_response.trace_context is None

        # Should still serialize/deserialize correctly
        for response in [text_response, tool_response, structured_response]:
            data = response.model_dump()
            assert 'trace_context' in data
            assert data['trace_context'] is None

            # Should be able to recreate from data
            recreated = type(response).model_validate(data)
            assert recreated.trace_context is None

    def test_token_summary_unchanged(self):
        """Test that TokenSummary class is unchanged."""
        # TokenSummary should NOT have trace_context
        summary = TokenSummary(
            input_tokens=10,
            output_tokens=5,
            duration_seconds=1.0,
        )

        # Should not have trace_context attribute
        assert not hasattr(summary, 'trace_context')

        # Should still work as before
        assert summary.total_tokens == 15
        assert summary.total_cost is None
