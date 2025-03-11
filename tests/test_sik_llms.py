"""Test public functions from sik_llms module."""
import os
import pytest
from sik_llms import (
    create_client,
    user_message,
    TextChunkEvent,
    TextResponse,
    Tool,
    RegisteredClients,
    ToolPredictionResponse,
    ToolPrediction,
)
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL


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
    async for response in client.stream(messages=[user_message("What is the capital of France?")]):
        if isinstance(response, TextChunkEvent):
            responses.append(response)

    assert len(responses) > 0
    assert 'Paris' in ''.join([response.content for response in responses])

    response = client(messages=[user_message("What is the capital of France?")])
    assert isinstance(response, TextResponse)
    assert 'Paris' in response.response


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
    async for response in client.stream(messages=[user_message("What is the capital of France?")]):
        if isinstance(response, TextChunkEvent):
            responses.append(response)

    assert len(responses) > 0
    assert 'Paris' in ''.join([response.content for response in responses])

    response = client(messages=[user_message("What is the capital of France?")])
    assert isinstance(response, TextResponse)
    assert 'Paris' in response.response


@pytest.mark.asyncio
class TestOpenAIFunctions:
    """Test the OpenAIFunctions class."""

    @pytest.mark.parametrize('is_async', [True, False])
    async def test_single_function_single_parameter__instantiate(
            self,
            simple_weather_tool: Tool,
            is_async: bool,
        ):
        """Test calling a simple function with one required parameter."""
        client = create_client(
            client_type=RegisteredClients.OPENAI_TOOLS,
            model_name=OPENAI_TEST_MODEL,
            tools=[simple_weather_tool],
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
        assert isinstance(response, ToolPredictionResponse)
        assert isinstance(response.tool_prediction, ToolPrediction)
        assert response.tool_prediction.name == "get_weather"
        assert "location" in response.tool_prediction.arguments
        assert "Paris" in response.tool_prediction.arguments["location"]
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
            simple_weather_tool: Tool,
            is_async: bool,
        ):
        """Test calling a simple function with one required parameter."""
        client = create_client(
            client_type=RegisteredClients.ANTHROPIC_TOOLS,
            model_name=ANTHROPIC_TEST_MODEL,
            tools=[simple_weather_tool],
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
        assert isinstance(response, ToolPredictionResponse)
        assert isinstance(response.tool_prediction, ToolPrediction)
        assert response.tool_prediction.name == "get_weather"
        assert "location" in response.tool_prediction.arguments
        assert "Paris" in response.tool_prediction.arguments["location"]
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
