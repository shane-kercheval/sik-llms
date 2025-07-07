"""Regression tests for telemetry integration."""
import os
from unittest.mock import patch, MagicMock
from sik_llms import create_client
from sik_llms.models_base import Client
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL


class TestTelemetryRegression:
    """Ensure telemetry doesn't break existing functionality."""

    def test_all_existing_functionality_telemetry_disabled(self):
        """Run key existing scenarios with telemetry disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            # Test basic client creation
            client = create_client(OPENAI_TEST_MODEL)
            assert client.tracer is None
            assert client.meter is None

            # Test all client types
            for model in [OPENAI_TEST_MODEL, ANTHROPIC_TEST_MODEL]:
                client = create_client(model)
                assert hasattr(client, 'model_name')
                assert client.model_name == model

                # Test that telemetry-related methods exist but don't break
                assert hasattr(client, 'tracer')
                assert hasattr(client, 'meter')
                assert hasattr(client, '_get_provider_name')

    def test_all_existing_functionality_telemetry_mocked(self):
        """Run key existing scenarios with telemetry enabled but mocked."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            with patch('sik_llms.models_base.get_tracer', return_value=None):
                with patch('sik_llms.models_base.get_meter', return_value=None):
                    # Same tests as above should still work
                    client = create_client(OPENAI_TEST_MODEL)
                    assert hasattr(client, 'model_name')
                    assert client.tracer is None
                    assert client.meter is None

    def test_reasoning_agent_with_telemetry_disabled(self):
        """Test ReasoningAgent specifically with telemetry disabled."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            from sik_llms import ReasoningAgent
            agent = ReasoningAgent(model_name=OPENAI_TEST_MODEL)
            assert agent.tracer is None
            assert agent.meter is None
            assert hasattr(agent, '_get_provider_name')

    def test_reasoning_agent_with_telemetry_mocked(self):
        """Test ReasoningAgent with telemetry enabled but mocked."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            with patch('sik_llms.models_base.get_tracer') as mock_tracer:
                with patch('sik_llms.models_base.get_meter') as mock_meter:
                    mock_tracer_instance = MagicMock()
                    mock_meter_instance = MagicMock()
                    mock_tracer.return_value = mock_tracer_instance
                    mock_meter.return_value = mock_meter_instance

                    from sik_llms import ReasoningAgent
                    agent = ReasoningAgent(model_name=OPENAI_TEST_MODEL)
                    assert agent.tracer is mock_tracer_instance
                    assert agent.meter is mock_meter_instance

    def test_client_methods_work_without_telemetry(self):
        """Test that all client methods work without telemetry."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = create_client(OPENAI_TEST_MODEL)

            # Test provider name method
            provider_name = client._get_provider_name()
            assert provider_name == "openai"

            # Test that methods can be called (even if they don't execute due to mocking)
            assert hasattr(client, 'run_async')
            assert hasattr(client, 'sample')
            assert hasattr(client, 'generate_multiple')
            assert hasattr(client, 'stream')

    def test_anthropic_client_without_telemetry(self):
        """Test Anthropic client works without telemetry."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            client = create_client(ANTHROPIC_TEST_MODEL)
            assert client.tracer is None
            assert client.meter is None
            provider_name = client._get_provider_name()
            assert provider_name == "anthropic"

    def test_token_summary_emit_metrics_no_meter(self):
        """Test TokenSummary.emit_metrics works with no meter."""
        from sik_llms.models_base import TokenSummary

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
        )

        # Should not raise exception
        summary.emit_metrics(None)
        summary.emit_metrics(None, {"model": "test"})

    def test_safe_span_works_with_none_tracer(self):
        """Test safe_span works with None tracer."""
        from sik_llms.telemetry import safe_span

        # Should return nullcontext and not raise exception
        with safe_span(None, "test_span"):
            pass

        with safe_span(None, "test_span", attributes={"key": "value"}):
            pass

    def test_existing_imports_still_work(self):
        """Test that all existing imports still work."""
        # Core imports
        from sik_llms import (
            create_client,
            Client,
            RegisteredClients,
            ModelProvider,
            ModelInfo,
            ImageContent,
            ImageSourceType,
            system_message,
            user_message,
            assistant_message,
            TextChunkEvent,
            ErrorEvent,
            InfoEvent,
            AgentEvent,
            ThinkingEvent,
            ThinkingChunkEvent,
            ToolPredictionEvent,
            ToolResultEvent,
            TextResponse,
            Parameter,
            Tool,
            ToolPrediction,
            ToolPredictionResponse,
            ToolChoice,
            StructuredOutputResponse,
            ReasoningEffort,
            ReasoningAgent,
            OpenAI,
            OpenAITools,
            Anthropic,
            AnthropicTools,
        )

        # Telemetry imports
        from sik_llms import (
            is_telemetry_enabled,
            get_tracer,
            get_meter,
            create_span_link,
        )

        # All should be importable
        assert all([
            create_client, Client, RegisteredClients, ModelProvider, ModelInfo,
            ImageContent, ImageSourceType, system_message, user_message, assistant_message,
            TextChunkEvent, ErrorEvent, InfoEvent, AgentEvent, ThinkingEvent,
            ThinkingChunkEvent, ToolPredictionEvent, ToolResultEvent, TextResponse,
            Parameter, Tool, ToolPrediction, ToolPredictionResponse, ToolChoice,
            StructuredOutputResponse, ReasoningEffort, ReasoningAgent,
            OpenAI, OpenAITools, Anthropic, AnthropicTools,
            is_telemetry_enabled, get_tracer, get_meter, create_span_link,
        ])

    def test_client_registration_still_works(self):
        """Test that client registration mechanism still works."""
        from sik_llms import RegisteredClients

        # Test that clients are still registered using enum values
        assert Client.is_registered(RegisteredClients.OPENAI)
        assert Client.is_registered(RegisteredClients.ANTHROPIC)
        assert Client.is_registered(RegisteredClients.OPENAI_TOOLS)
        assert Client.is_registered(RegisteredClients.ANTHROPIC_TOOLS)
        assert Client.is_registered(RegisteredClients.REASONING_AGENT)

    def test_model_instantiation_still_works(self):
        """Test that model instantiation still works."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            from sik_llms import RegisteredClients

            # Test direct instantiation
            openai_client = Client.instantiate(RegisteredClients.OPENAI, OPENAI_TEST_MODEL)
            assert openai_client.model_name == OPENAI_TEST_MODEL

            anthropic_client = Client.instantiate(RegisteredClients.ANTHROPIC, ANTHROPIC_TEST_MODEL)  # noqa: E501
            assert anthropic_client.model_name == ANTHROPIC_TEST_MODEL

    def test_pydantic_utilities_still_work(self):
        """Test that Pydantic utilities still work."""
        from sik_llms.models_base import pydantic_model_to_parameters, pydantic_model_to_tool
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int

        # Should not raise exceptions
        params = pydantic_model_to_parameters(TestModel)
        assert len(params) == 2

        tool = pydantic_model_to_tool(TestModel)
        assert tool.name == "TestModel"
        assert len(tool.parameters) == 2

    def test_telemetry_env_var_variations(self):
        """Test various environment variable settings don't break functionality."""
        test_cases = [
            {"OTEL_SDK_DISABLED": "true"},
            {"OTEL_SDK_DISABLED": "false"},
            {"OTEL_SDK_DISABLED": "TRUE"},
            {"OTEL_SDK_DISABLED": "FALSE"},
            {"OTEL_SDK_DISABLED": "1"},
            {"OTEL_SDK_DISABLED": "0"},
            {"OTEL_SDK_DISABLED": "yes"},
            {"OTEL_SDK_DISABLED": "no"},
            {},  # No environment variable set
        ]

        for env_vars in test_cases:
            with patch.dict(os.environ, env_vars, clear=True):
                with patch('sik_llms.telemetry.get_tracer', return_value=None):
                    with patch('sik_llms.telemetry.get_meter', return_value=None):
                        # Should not raise exceptions
                        client = create_client(OPENAI_TEST_MODEL)
                        assert hasattr(client, 'model_name')
                        assert client.model_name == OPENAI_TEST_MODEL


class TestTelemetryPerformanceRegression:
    """Test that telemetry doesn't significantly impact performance when disabled."""

    def test_client_creation_performance_disabled(self):
        """Test client creation performance with telemetry disabled."""
        import time

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            start_time = time.time()

            # Create multiple clients
            for _ in range(10):
                client = create_client(OPENAI_TEST_MODEL)
                assert client.tracer is None
                assert client.meter is None

            end_time = time.time()
            duration = end_time - start_time

            # Should be very fast (under 1 second for 10 clients)
            assert duration < 1.0

    def test_safe_span_performance_no_tracer(self):
        """Test safe_span performance with no tracer."""
        import time
        from sik_llms.telemetry import safe_span

        start_time = time.time()

        # Use safe_span many times with no tracer
        for _ in range(1000):
            with safe_span(None, "test_span"):
                pass

        end_time = time.time()
        duration = end_time - start_time

        # Should be very fast (under 0.1 seconds for 1000 operations)
        assert duration < 0.1

    def test_emit_metrics_performance_no_meter(self):
        """Test emit_metrics performance with no meter."""
        import time
        from sik_llms.models_base import TokenSummary

        summary = TokenSummary(
            input_tokens=100,
            output_tokens=50,
            duration_seconds=2.5,
        )

        start_time = time.time()

        # Call emit_metrics many times with no meter
        for _ in range(1000):
            summary.emit_metrics(None)

        end_time = time.time()
        duration = end_time - start_time

        # Should be very fast (under 0.1 seconds for 1000 operations)
        assert duration < 0.1


class TestTelemetryBackwardCompatibility:
    """Test backward compatibility with existing code patterns."""

    def test_existing_client_usage_patterns(self):
        """Test that existing client usage patterns still work."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            # Pattern 1: Direct client creation
            client = create_client(OPENAI_TEST_MODEL)
            assert isinstance(client, Client)

            # Pattern 2: Model-specific creation
            from sik_llms import OpenAI, Anthropic
            openai_client = OpenAI(OPENAI_TEST_MODEL)
            anthropic_client = Anthropic(ANTHROPIC_TEST_MODEL)

            assert openai_client.model_name == OPENAI_TEST_MODEL
            assert anthropic_client.model_name == ANTHROPIC_TEST_MODEL

            # Pattern 3: Registry-based creation
            registry_client = Client.instantiate("OpenAI", OPENAI_TEST_MODEL)
            assert registry_client.model_name == OPENAI_TEST_MODEL

    def test_existing_message_utilities(self):
        """Test that message utility functions still work."""
        from sik_llms import system_message, user_message, assistant_message

        sys_msg = system_message("You are a helpful assistant")
        user_msg = user_message("Hello")
        asst_msg = assistant_message("Hi there!")

        assert sys_msg["role"] == "system"
        assert user_msg["role"] == "user"
        assert asst_msg["role"] == "assistant"

        assert sys_msg["content"] == "You are a helpful assistant"
        assert user_msg["content"] == "Hello"
        assert asst_msg["content"] == "Hi there!"

    def test_existing_model_info_access(self):
        """Test that model info access still works."""
        from sik_llms import SUPPORTED_OPENAI_MODELS, SUPPORTED_ANTHROPIC_MODELS

        assert OPENAI_TEST_MODEL in SUPPORTED_OPENAI_MODELS
        assert ANTHROPIC_TEST_MODEL in SUPPORTED_ANTHROPIC_MODELS

        # Test model info structure
        gpt_info = SUPPORTED_OPENAI_MODELS[OPENAI_TEST_MODEL]
        claude_info = SUPPORTED_ANTHROPIC_MODELS[ANTHROPIC_TEST_MODEL]

        assert hasattr(gpt_info, 'model')
        assert hasattr(gpt_info, 'provider')
        assert hasattr(claude_info, 'model')
        assert hasattr(claude_info, 'provider')

    def test_existing_tool_functionality(self):
        """Test that tool-related functionality still works."""
        from sik_llms import Tool, Parameter, ToolChoice

        # Create a parameter
        param = Parameter(
            name="query",
            param_type=str,
            required=True,
            description="Search query",
        )

        # Create a tool
        tool = Tool(
            name="search",
            parameters=[param],
            description="Search for information",
        )

        assert tool.name == "search"
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "query"

        # Test tool choice enum
        assert ToolChoice.AUTO
        assert ToolChoice.REQUIRED

    def test_provider_detection_backward_compatibility(self):
        """Test that the new provider detection doesn't break existing functionality."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
            with patch('sik_llms.telemetry.get_tracer', return_value=MagicMock()):
                with patch('sik_llms.telemetry.get_meter', return_value=MagicMock()):
                    # Should work the same as before for new users
                    client = create_client(OPENAI_TEST_MODEL)
                    assert client.tracer is not None  # Should auto-configure

        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            # Should still respect disabled state
            client = create_client(OPENAI_TEST_MODEL)
            assert client.tracer is None
