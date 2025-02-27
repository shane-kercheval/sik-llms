"""Test public functions from sik_llms module."""
import os
import pytest
from sik_llms import (
    create_client,
    user_message,
    ChatChunkResponse,
    ChatResponseSummary,
    Function,
    RegisteredClients,
    FunctionCallResponse,
    FunctionCallResult,
)


OPENAI_TEST_MODEL = 'gpt-4o-mini'
ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'


@pytest.mark.asyncio
async def test__create_client__openai() -> None:
    """Tests that model can be called from create_client."""
    client = create_client(model_name=OPENAI_TEST_MODEL)
    assert client.client is not None
    assert client.client.api_key
    assert client.client.api_key != 'None'
    assert client.client.base_url
    assert client.model == OPENAI_TEST_MODEL

    responses = []
    async for response in client.run_async(messages=[user_message("What is the capital of France?")]):  # noqa: E501
        if isinstance(response, ChatChunkResponse):
            responses.append(response)

    assert len(responses) > 0
    assert 'Paris' in ''.join([response.content for response in responses])

    response = client(messages=[user_message("What is the capital of France?")])
    assert isinstance(response, ChatResponseSummary)
    assert 'Paris' in response.content


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__create_client__anthropic() -> None:
    """Tests that model can be called from create_client."""
    client = create_client(model_name=ANTHROPIC_TEST_MODEL)
    assert client.client is not None
    assert client.client.api_key
    assert client.client.api_key != 'None'
    assert client.client.base_url
    assert client.model == ANTHROPIC_TEST_MODEL

    responses = []
    async for response in client.run_async(messages=[user_message("What is the capital of France?")]):  # noqa: E501
        if isinstance(response, ChatChunkResponse):
            responses.append(response)

    assert len(responses) > 0
    assert 'Paris' in ''.join([response.content for response in responses])

    response = client(messages=[user_message("What is the capital of France?")])
    assert isinstance(response, ChatResponseSummary)
    assert 'Paris' in response.content


@pytest.mark.asyncio
class TestOpenAIFunctions:
    """Test the OpenAIFunctions class."""

    @pytest.mark.parametrize('is_async', [True, False])
    async def test_single_function_single_parameter__instantiate(
            self,
            simple_weather_function: Function,
            is_async: bool,
        ):
        """Test calling a simple function with one required parameter."""
        client = create_client(
            client_type=RegisteredClients.OPENAI_FUNCTIONS,
            model_name=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
        )
        if is_async:
            response = await client.run_async(
                messages=[
                    user_message("What's the weather like in Paris?"),
                ],
            )
        else:
            response = client(
                messages=[
                    user_message("What's the weather like in Paris?"),
                ],
            )
        assert isinstance(response, FunctionCallResponse)
        assert isinstance(response.function_call, FunctionCallResult)
        assert response.function_call.name == "get_weather"
        assert "location" in response.function_call.arguments
        assert "Paris" in response.function_call.arguments["location"]
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
class TestAnthropicFunctions:
    """Test the AnthropicFunctions class."""

    @pytest.mark.parametrize('is_async', [True, False])
    async def test_single_function_single_parameter__instantiate(
            self,
            simple_weather_function: Function,
            is_async: bool,
        ):
        """Test calling a simple function with one required parameter."""
        client = create_client(
            client_type=RegisteredClients.ANTHROPIC_FUNCTIONS,
            model_name=ANTHROPIC_TEST_MODEL,
            functions=[simple_weather_function],
        )
        if is_async:
            response = await client.run_async(
                messages=[
                    user_message("What's the weather like in Paris?"),
                ],
            )
        else:
            response = client(
                messages=[
                    user_message("What's the weather like in Paris?"),
                ],
            )
        assert isinstance(response, FunctionCallResponse)
        assert isinstance(response.function_call, FunctionCallResult)
        assert response.function_call.name == "get_weather"
        assert "location" in response.function_call.arguments
        assert "Paris" in response.function_call.arguments["location"]
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
