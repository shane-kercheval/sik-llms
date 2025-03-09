"""Test the OpenAI Wrapper."""
import asyncio
import time
import pytest
from dotenv import load_dotenv
from sik_llms import (
    Client,
    ReasoningEffort,
    create_client,
    system_message,
    user_message,
    Tool,
    Parameter,
    RegisteredClients,
    OpenAI,
    TextChunkEvent,
    ResponseSummary,
    OpenAITools,
)
from tests.conftest import OPENAI_TEST_MODEL, OPENAI_TEST_REASONING_MODEL

load_dotenv()


def test__registration__openai():
    assert Client.is_registered(RegisteredClients.OPENAI)


def test__registration__openai_tools():
    assert Client.is_registered(RegisteredClients.OPENAI_TOOLS)


@pytest.mark.asyncio
class TestOpenAIRegistration:
    """Test the OpenAI Completion Wrapper registration."""

    async def test__openai__instantiate(self):
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name=OPENAI_TEST_MODEL,
        )
        assert isinstance(client, OpenAI)
        assert client.model == OPENAI_TEST_MODEL
        assert client.client is not None
        assert client.client.api_key
        assert client.client.api_key != 'None'
        assert client.client.base_url

        # call async/stream method
        responses = []
        async for response in client.run_async(messages=[user_message("What is the capital of France?")]):  # noqa: E501
            if isinstance(response, TextChunkEvent):
                responses.append(response)

        assert len(responses) > 0
        assert 'Paris' in ''.join([response.content for response in responses])

        # call non-async method
        response = client(messages=[user_message("What is the capital of France?")])
        assert isinstance(response, ResponseSummary)
        assert 'Paris' in response.response

    async def test__openai__compatible_server_instantiate___parameters__missing_server_url(self):
        # This base_url is required for openai-compatible-server
        with pytest.raises(ValueError):  # noqa: PT011
            _ = Client.instantiate(
                client_type=RegisteredClients.OPENAI,
                model_name='openai-compatible-server',
            )

    async def test__openai__compatible_server_instantiate___parameters__missing_max_tokens(self):
        model_config = {
            'server_url': 'http://localhost:8000',
            'temperature': 0.5,
        }
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name='openai-compatible-server',
            **model_config,
        )
        assert isinstance(client, OpenAI)
        assert client.model == 'openai-compatible-server'
        assert client.model_parameters['temperature'] == 0.5
        # we need to set the max_tokens parameter to -1 to avoid it sending just 1 token
        assert client.model_parameters['max_tokens'] is not None

    async def test__openai__compatible_server_instantiate___parameters_max_tokens(self):
        model_config = {
            'server_url': 'http://localhost:8000',
            'max_tokens': 10,
        }
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name='openai-compatible-server',
            **model_config,
        )
        assert isinstance(client, OpenAI)
        assert client.model == 'openai-compatible-server'
        assert client.model_parameters['max_tokens'] == 10

    async def test__openai__tools_instantiate(self):
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI_TOOLS,
            model_name=OPENAI_TEST_MODEL,
            tools=[
                Tool(
                    name='get_weather',
                    description='Get the weather for a location.',
                    parameters=[
                        Parameter(
                            name='location',
                            param_type=str,
                            required=True,
                            description='The city and country for weather info.',
                        ),
                    ],
                ),
            ],
        )
        assert isinstance(client, OpenAITools)
        assert client.model == OPENAI_TEST_MODEL
        assert client.client is not None
        assert client.client.api_key
        assert client.client.api_key != 'None'
        assert client.client.base_url
        assert client


class TestReasoningModelsDoNotSetCertainParameters:
    """When using reasoning models, certain parameters should not be set."""

    def test__check_parameters_for_none_reasoning(self):
        client = OpenAI(
            model_name=OPENAI_TEST_MODEL,
            reasoning_effort=None,
            temperature=0.5,
            top_p=0.9,
        )
        assert client.reasoning_effort is None
        assert 'temperature' in client.model_parameters
        assert 'top_p' in client.model_parameters
        assert client.log_probs is True

    def test__does_not_set_params__reasoning_model(self):
        client = OpenAI(
            model_name=OPENAI_TEST_REASONING_MODEL,
            # reasoning model but no reasoning effort set
            reasoning_effort=None,
            temperature=0.5,
            top_p=0.9,
        )
        assert client.reasoning_effort is None
        assert 'temperature' not in client.model_parameters
        assert 'top_p' not in client.model_parameters
        assert client.log_probs is False

    @pytest.mark.parametrize('reasoning_effort', [None, ReasoningEffort.LOW])
    def test__does_not_set_params__reasoning_effort(self, reasoning_effort: ReasoningEffort | None):  # noqa: E501
        client = OpenAI(
            model_name=OPENAI_TEST_REASONING_MODEL if reasoning_effort else OPENAI_TEST_MODEL,
            reasoning_effort=reasoning_effort,
            temperature=0.5,
            top_p=0.9,
        )
        assert client.reasoning_effort == reasoning_effort
        assert 'temperature' not in client.model_parameters if reasoning_effort else 'temperature' in client.model_parameters  # noqa: E501
        assert 'top_p' not in client.model_parameters if reasoning_effort else 'top_p' in client.model_parameters  # noqa: E501
        assert client.log_probs if not reasoning_effort else not client.log_probs


@pytest.mark.asyncio
class TestOpenAI:
    """Test the OpenAI Completion Wrapper."""

    async def test__async_openai(self):
        # Create an instance of the wrapper
        client = OpenAI(model_name=OPENAI_TEST_MODEL)
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]

        async def run_model():  # noqa: ANN202
            chunks = []
            summary = None
            try:
                async for response in client.run_async(messages=messages):
                    if isinstance(response, TextChunkEvent):
                        chunks.append(response)
                    elif isinstance(response, ResponseSummary):
                        summary = response
                return chunks, summary
            except Exception:
                return [], None

        results = await asyncio.gather(*(run_model() for _ in range(10)))
        passed_tests = []
        for chunks, summary in results:
            assert all(chunk.logprob is not None for chunk in chunks)
            response = ''.join([chunk.content for chunk in chunks])
            passed_tests.append(
                'Paris' in response
                and isinstance(summary, ResponseSummary)
                and summary.input_tokens > 0
                and summary.output_tokens > 0
                and summary.input_cost > 0
                and summary.output_cost > 0
                and summary.duration_seconds > 0,
            )
        assert sum(passed_tests) / len(passed_tests) >= 0.9, f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."  # noqa: E501

    async def test_concurrent_performance(self):
        # This test will use the actual API, so consider using it sparingly
        client = OpenAI(model_name=OPENAI_TEST_MODEL)
        num_requests = 20
        messages = [{"role": "user", "content": "Say hello in one word."}]

        async def execute_request(i: int):  # noqa: ANN202
            start = time.time()
            result = None
            async for chunk in client.run_async(messages=messages):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    result = chunk
            end = time.time()
            return {"index": i, "time": end - start, "result": result}

        # Execute all requests asynchronously/concurrently
        start_time = time.time()
        tasks = [execute_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        actual_time = time.time() - start_time

        times = [r["time"] for r in results]
        sequential_time = sum(times)

        assert len(results) == num_requests
        # The actual time should be significantly less than sequential execution
        assert actual_time < (sequential_time / 2)


@pytest.mark.asyncio
class TestOpenAIReasoning:
    """Test the OpenAI Reasoning / ReasoningEffort."""

    async def test__openai__reasoning(self):
        client = create_client(
            model_name=OPENAI_TEST_REASONING_MODEL,
            reasoning_effort=ReasoningEffort.LOW,
        )
        response = client(messages=[user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")])
        assert isinstance(response, ResponseSummary)
        assert '45' in response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.duration_seconds > 0
        response = client(messages=[user_message("What is the capital of France?")])
        assert 'Paris' in response.response

    async def test__openai__test_reasoning_model__without_reasonsing_effort_set(self):
        """
        Logprobs are not supported for reasoning models so let's test that we don't set it. In
        general, let's test that the reasoning model works without setting reasonsing.
        """
        client = create_client(
            model_name=OPENAI_TEST_REASONING_MODEL,
            reasoning_effort=None,
        )
        response = client(messages=[user_message("What is the capital of France?")])
        assert isinstance(response, ResponseSummary)
        assert 'Paris' in response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.duration_seconds > 0
