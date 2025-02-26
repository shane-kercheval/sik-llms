"""Test the Anthropic Wrapper."""
import asyncio
import os
import pytest
from dotenv import load_dotenv
from sik_llms import (
    Client,
    system_message,
    user_message,
    RegisteredClients,
    Anthropic,
    ChatChunkResponse,
    ChatResponseSummary,
)
load_dotenv()

ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__async_anthropic_completion_wrapper_call():
    """Test the Anthropic wrapper with multiple concurrent calls."""
    # Create an instance of the wrapper
    client = Anthropic(
        model_name=ANTHROPIC_TEST_MODEL,
        max_tokens=100,
    )
    messages = [
        system_message("You are a helpful assistant."),
        user_message("What is the capital of France?"),
    ]

    async def run_model():  # noqa: ANN202
        chunks = []
        summary = None
        try:
            async for response in client.run_async(messages=messages):
                if isinstance(response, ChatChunkResponse):
                    chunks.append(response)
                elif isinstance(response, ChatResponseSummary):
                    summary = response
            return chunks, summary
        except Exception:
            return [], None

    results = await asyncio.gather(*(run_model() for _ in range(10)))
    passed_tests = []

    for chunks, summary in results:
        response = ''.join([chunk.content for chunk in chunks])
        passed_tests.append(
            'Paris' in response
            and isinstance(summary, ChatResponseSummary)
            and summary.total_input_tokens > 0
            and summary.total_output_tokens > 0
            and summary.total_input_cost > 0
            and summary.total_output_cost > 0
            and summary.duration_seconds > 0,
        )

    assert sum(passed_tests) / len(passed_tests) >= 0.9, (
        f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."
    )


def test__Anthropic_registration():
    assert Client.is_registered(RegisteredClients.ANTHROPIC)


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
def test__Anthropic__instantiate():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
    )
    assert isinstance(model, Anthropic)
    assert model.model == ANTHROPIC_TEST_MODEL
    assert model.client is not None
    assert model.client.api_key
    assert model.client.api_key != 'None'


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__instantiate__run_async():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
    )
    responses = []
    async for response in model.run_async(messages=[user_message("What is the capital of France?")]):  # noqa: E501
        if isinstance(response, ChatChunkResponse):
            responses.append(response)

    assert len(responses) > 0
    assert 'Paris' in ''.join([response.content for response in responses])



@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
def test__Anthropic_instantiate___parameters():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
        temperature=0.5,
        max_tokens=100,
    )
    assert isinstance(model, Anthropic)
    assert model.model == ANTHROPIC_TEST_MODEL
    assert model.model_parameters['temperature'] == 0.5
    assert model.model_parameters['max_tokens'] == 100
