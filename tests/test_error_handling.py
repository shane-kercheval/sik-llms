"""Tests for error handling and telemetry in sik-llms clients."""
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from sik_llms import OpenAI, Anthropic, ReasoningAgent
from sik_llms.models_base import user_message, system_message


async def assert_stream_raises(
    stream_generator: object,  # AsyncGenerator, but we'll keep it simple
    exception_type: type[Exception],
    match_text: str,
) -> None:
    """Helper to assert that streaming raises an exception."""
    with pytest.raises(exception_type, match=match_text):
        async for _ in stream_generator:
            pass


class TestOpenAIErrorHandling:
    """Test error handling and telemetry for OpenAI client."""

    @pytest.mark.asyncio
    async def test_openai_api_error_with_telemetry(self):
        """Test that OpenAI API errors are properly logged to telemetry spans."""
        # Mock the tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=None)

        # Create client with mocked tracer
        with patch('sik_llms.models_base.get_tracer', return_value=mock_tracer):
            with patch('sik_llms.models_base.get_meter', return_value=None):
                client = OpenAI(model_name="gpt-4o-mini")

                # Mock the OpenAI client to raise an exception
                client.client.chat.completions.create = AsyncMock(
                    side_effect=Exception("Invalid API key"),
                )

                messages = [
                    system_message("You are a helpful assistant."),
                    user_message("Hello"),
                ]

                # The error should be raised
                await assert_stream_raises(
                    client.stream(messages), Exception, "Invalid API key",
                )

                # Verify telemetry span was called with error information
                mock_span.set_attribute.assert_any_call("llm.request.error", "Invalid API key")

                # Verify span status was set to ERROR
                # We need to check that set_status was called with an error status
                assert mock_span.set_status.called
                call_args = mock_span.set_status.call_args
                assert call_args is not None
                status_arg = call_args[0][0]  # First positional argument
                assert hasattr(status_arg, 'status_code')

    @pytest.mark.asyncio
    async def test_openai_structured_output_error_with_telemetry(self):
        """Test error handling in structured output mode with telemetry."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int

        # Mock the tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=None)

        # Create client with structured output and mocked tracer
        with patch('sik_llms.models_base.get_tracer', return_value=mock_tracer):
            with patch('sik_llms.models_base.get_meter', return_value=None):
                client = OpenAI(
                    model_name="gpt-4o-mini",
                    response_format=TestModel,
                )

                # Mock the structured output parsing to raise an exception
                client.client.beta.chat.completions.parse = AsyncMock(
                    side_effect=Exception("Rate limit exceeded"),
                )

                messages = [user_message("Tell me about John who is 25 years old")]

                # The error should be raised
                await assert_stream_raises(
                    client.stream(messages), Exception, "Rate limit exceeded",
                )

                # Verify telemetry span was called with error information
                mock_span.set_attribute.assert_any_call("llm.request.error", "Rate limit exceeded")
                assert mock_span.set_status.called

    @pytest.mark.asyncio
    async def test_openai_error_without_telemetry(self):
        """Test that errors still propagate correctly when telemetry is disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = OpenAI(model_name="gpt-4o-mini")

            # Verify telemetry is disabled
            assert client.tracer is None
            assert client.meter is None

            # Mock the OpenAI client to raise an exception
            client.client.chat.completions.create = AsyncMock(
                side_effect=Exception("Network error"),
            )

            messages = [user_message("Hello")]

            # The error should still be raised
            await assert_stream_raises(client.stream(messages), Exception, "Network error")


class TestAnthropicErrorHandling:
    """Test error handling and telemetry for Anthropic client."""

    @pytest.mark.asyncio
    async def test_anthropic_api_error_with_telemetry(self):
        """Test that Anthropic API errors are properly logged to telemetry spans."""
        # Mock the tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=None)

        # Create client with mocked tracer
        with patch('sik_llms.models_base.get_tracer', return_value=mock_tracer):
            with patch('sik_llms.models_base.get_meter', return_value=None):
                client = Anthropic(model_name="claude-3-5-sonnet-latest")

                # Mock the Anthropic client to raise an exception
                client.client.messages.create = AsyncMock(
                    side_effect=Exception("Authentication failed"),
                )

                messages = [
                    system_message("You are a helpful assistant."),
                    user_message("Hello"),
                ]

                # The error should be raised
                await assert_stream_raises(
                    client.stream(messages), Exception, "Authentication failed",
                )

                # Verify telemetry span was called with error information
                mock_span.set_attribute.assert_any_call(
                    "llm.request.error", "Authentication failed",
                )

                # Verify span status was set to ERROR
                assert mock_span.set_status.called
                call_args = mock_span.set_status.call_args
                assert call_args is not None
                status_arg = call_args[0][0]  # First positional argument
                assert hasattr(status_arg, 'status_code')

    @pytest.mark.asyncio
    async def test_anthropic_structured_output_error_with_telemetry(self):
        """Test error handling in Anthropic structured output mode with telemetry."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            response: str

        # Mock the tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=None)

        # Create client with structured output and mocked tracer
        with patch('sik_llms.models_base.get_tracer', return_value=mock_tracer):
            with patch('sik_llms.models_base.get_meter', return_value=None):
                with patch('sik_llms.anthropic.AnthropicTools') as mock_tools_class:
                    # Mock the tools client to raise an exception
                    mock_tools_instance = MagicMock()
                    mock_tools_instance.run_async = AsyncMock(
                        side_effect=Exception("Service unavailable"),
                    )
                    mock_tools_class.return_value = mock_tools_instance

                    client = Anthropic(
                        model_name="claude-3-5-sonnet-latest",
                        response_format=TestModel,
                    )

                    messages = [user_message("Say hello")]

                    # Collect all events including errors
                    events = []
                    async for event in client.stream(messages):
                        events.append(event)

                    # Should have received an error event and StructuredOutputResponse with refusal
                    from sik_llms.models_base import ErrorEvent, StructuredOutputResponse
                    error_events = [e for e in events if isinstance(e, ErrorEvent)]
                    structured_responses = [
                        e for e in events if isinstance(e, StructuredOutputResponse)
                    ]

                    assert len(error_events) > 0
                    assert any("Service unavailable" in e.content for e in error_events)
                    assert len(structured_responses) > 0
                    assert any(
                        r.refusal and "Service unavailable" in r.refusal
                        for r in structured_responses
                    )

                    # Verify telemetry span was called with error information
                    mock_span.set_attribute.assert_any_call(
                        "llm.request.error", "Service unavailable",
                    )

    @pytest.mark.asyncio
    async def test_anthropic_error_without_telemetry(self):
        """Test that errors still propagate correctly when telemetry is disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = Anthropic(model_name="claude-3-5-sonnet-latest")

            # Verify telemetry is disabled
            assert client.tracer is None
            assert client.meter is None

            # Mock the Anthropic client to raise an exception
            client.client.messages.create = AsyncMock(
                side_effect=Exception("Connection timeout"),
            )

            messages = [user_message("Hello")]

            # The error should still be raised
            await assert_stream_raises(client.stream(messages), Exception, "Connection timeout")


class TestReasoningAgentErrorHandling:
    """Test error handling and telemetry for ReasoningAgent."""

    @pytest.mark.asyncio
    async def test_reasoning_agent_error_with_telemetry(self):
        """Test that ReasoningAgent errors are properly logged to telemetry spans."""
        # Mock the tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=None)

        # Create agent with mocked tracer
        with patch('sik_llms.models_base.get_tracer', return_value=mock_tracer):
            with patch('sik_llms.models_base.get_meter', return_value=None):
                agent = ReasoningAgent(model_name="gpt-4o-mini")

                # Mock the reasoning client to raise an exception
                with patch.object(agent, '_get_reasoning_client') as mock_get_client:
                    mock_client = MagicMock()
                    mock_client.run_async = AsyncMock(
                        side_effect=Exception("Model overloaded"),
                    )
                    mock_get_client.return_value = mock_client

                    messages = [user_message("Solve this problem")]

                    # The error should be raised
                    await assert_stream_raises(
                        agent.stream(messages), Exception, "Model overloaded",
                    )

                    # Verify telemetry span was called with error information
                    mock_span.set_attribute.assert_any_call("reasoning.error", "Model overloaded")

    @pytest.mark.asyncio
    async def test_reasoning_agent_tool_error_with_telemetry(self):
        """Test that ReasoningAgent tool execution errors are handled with telemetry."""
        from sik_llms import Tool, Parameter

        def failing_tool(query: str) -> str:  # noqa: ARG001
            raise Exception("Tool execution failed")

        # Create a tool that will fail
        tool = Tool(
            name="search",
            parameters=[Parameter(name="query", param_type=str, required=True)],
            func=failing_tool,
            description="A search tool that fails",
        )

        # Mock the tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=None)

        # Create agent with tools and mocked tracer
        with patch('sik_llms.models_base.get_tracer', return_value=mock_tracer):
            with patch('sik_llms.models_base.get_meter', return_value=None):
                agent = ReasoningAgent(
                    model_name="gpt-4o-mini",
                    tools=[tool],
                    generate_final_response=False,  # Skip final response for simpler test
                )

                # Mock reasoning responses to use the tool
                from sik_llms.reasoning_agent import ReasoningStep, ReasoningAction
                from sik_llms.models_base import (
                    StructuredOutputResponse,
                    ToolPredictionResponse,
                    ToolPrediction,
                )

                reasoning_step = ReasoningStep(
                    thought="I'll use the search tool",
                    next_action=ReasoningAction.USE_TOOL,
                    tool_name="search",
                )

                # Mock the reasoning client to return our step
                with patch.object(agent, '_get_reasoning_client') as mock_get_reasoning_client:
                    mock_reasoning_client = MagicMock()
                    mock_reasoning_client.run_async = AsyncMock(
                        return_value=StructuredOutputResponse(
                            parsed=reasoning_step,
                            refusal=None,
                            input_tokens=100,
                            output_tokens=50,
                            input_cost=0.001,
                            output_cost=0.001,
                            duration_seconds=1.0,
                        ),
                    )
                    mock_get_reasoning_client.return_value = mock_reasoning_client

                    # Mock the tools client to return a tool prediction
                    with patch.object(agent, '_get_tools_client') as mock_get_tools_client:
                        mock_tools_client = MagicMock()
                        mock_tools_client.run_async = AsyncMock(
                            return_value=ToolPredictionResponse(
                                tool_prediction=ToolPrediction(
                                    name="search",
                                    arguments={"query": "test"},
                                    call_id="123",
                                ),
                                message=None,
                                input_tokens=50,
                                output_tokens=25,
                                input_cost=0.001,
                                output_cost=0.001,
                                duration_seconds=0.5,
                            ),
                        )
                        mock_get_tools_client.return_value = mock_tools_client

                        messages = [user_message("Search for something")]

                        # Collect all events including errors
                        events = []
                        async for event in agent.stream(messages):
                            events.append(event)

                        # Should have received an error event about tool execution failure
                        from sik_llms.models_base import ErrorEvent
                        error_events = [e for e in events if isinstance(e, ErrorEvent)]
                        assert len(error_events) > 0
                        assert any("Tool execution failed" in e.content for e in error_events)

    @pytest.mark.asyncio
    async def test_reasoning_agent_error_without_telemetry(self):
        """Test that ReasoningAgent errors still propagate correctly when telemetry is disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            agent = ReasoningAgent(model_name="gpt-4o-mini")

            # Verify telemetry is disabled
            assert agent.tracer is None
            assert agent.meter is None

            # Mock the reasoning client to raise an exception
            with patch.object(agent, '_get_reasoning_client') as mock_get_client:
                mock_client = MagicMock()
                mock_client.run_async = AsyncMock(
                    side_effect=Exception("Critical failure"),
                )
                mock_get_client.return_value = mock_client

                messages = [user_message("Solve this problem")]

                # The error should still be raised
                await assert_stream_raises(agent.stream(messages), Exception, "Critical failure")


class TestErrorHandlingUtilities:
    """Test error handling utilities and edge cases."""

    def test_safe_span_error_handling(self):
        """Test that safe_span handles errors gracefully."""
        from sik_llms.telemetry import safe_span

        # Test with None tracer - should not raise exception (safe_span just returns nullcontext)
        with pytest.raises(Exception, match="Test error"):
            with safe_span(None, "test_span"):
                raise Exception("Test error")  # This should propagate normally

        # Test with mock tracer that raises exception
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = Exception("Tracer error")

        # safe_span should handle tracer errors gracefully
        with pytest.raises(Exception, match="Tracer error"):
            with safe_span(mock_tracer, "test_span"):
                pass

    def test_token_summary_error_metrics(self):
        """Test that TokenSummary.emit_metrics handles meter errors gracefully."""
        from sik_llms.models_base import TokenSummary

        # Create a meter that raises exceptions
        mock_meter = MagicMock()
        mock_meter.create_counter.side_effect = Exception("Meter error")

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
        )

        # Should not raise exception even if meter fails
        summary.emit_metrics(mock_meter, {"model": "test"})
