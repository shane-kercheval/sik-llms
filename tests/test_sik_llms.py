"""Test public functions from sik_llms module."""
import os
from time import perf_counter
import pytest
from sik_llms import (
    create_client,
    user_message,
    TextChunkEvent,
    TextResponse,
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


@pytest.mark.asyncio
async def test__run_async() -> None:
    """Tests that model can be called from run_async."""
    client = create_client(model_name=OPENAI_TEST_MODEL)
    response = await client.run_async(messages=[user_message("What is the capital of France?")])
    assert isinstance(response, TextResponse)
    assert response.response
    assert 'Paris' in response.response
    assert response.total_tokens > 0
    assert response.total_cost > 0
    assert response.duration_seconds > 0


@pytest.mark.asyncio
async def test__sample() -> None:
    """Tests that model can be called from run_async."""
    client = create_client(model_name=OPENAI_TEST_MODEL, temperature=0.5)
    start_time = perf_counter()
    responses = await client.sample(
        messages=[user_message("What is the capital of France?")],
        n=3,
    )
    total_duration = perf_counter() - start_time
    assert isinstance(responses, list)
    assert len(responses) == 3
    assert isinstance(responses[0], TextResponse)
    for response in responses:
        assert response.response
        assert 'Paris' in response.response
        assert response.total_tokens > 0
        assert response.total_cost > 0
        assert response.duration_seconds > 0
    assert total_duration > 0
    assert total_duration < sum(response.duration_seconds for response in responses), \
        "Total duration should be less than the sum of individual durations because of concurrency"


@pytest.mark.asyncio
async def test__generate_multiple__requires_list_of_list() -> None:
    client = create_client(model_name=OPENAI_TEST_MODEL, temperature=0.1)
    with pytest.raises(TypeError):
        _ = await client.generate_multiple(
            # This should be a list of lists, but we are passing a single list
            messages=[user_message("What is the capital of France? Return only the city name.")],
        )

@pytest.mark.asyncio
async def test__generate_multiple() -> None:
    """generate_multiple with `sample_n` of 1 should return flattened list of responses."""
    client = create_client(model_name=OPENAI_TEST_MODEL, temperature=0.1)
    start_time = perf_counter()
    responses = await client.generate_multiple(
        messages=[
            [user_message("What is the capital of France? Return only the city name.")],
            [user_message("What is the capital of Italy? Return only the city name.")],
        ],
    )
    total_duration = perf_counter() - start_time
    assert isinstance(responses, list)
    assert len(responses) == 2
    assert isinstance(responses[0], TextResponse)
    for i, response in enumerate(responses):
        assert response.response
        if i == 0:
            assert 'Paris' in response.response
        else:
            assert 'Rome' in response.response
        assert response.total_tokens > 0
        assert response.total_cost > 0
        assert response.duration_seconds > 0
    assert total_duration > 0
    assert total_duration < sum(response.duration_seconds for response in responses), \
        "Total duration should be less than the sum of individual durations because of concurrency"


@pytest.mark.asyncio
async def test__generate_multiple__sample_n() -> None:
    """generate_multiple with `sample_n` greater than 1 should result in list of lists."""
    client = create_client(model_name=OPENAI_TEST_MODEL, temperature=0.1)
    start_time = perf_counter()
    # responses should be a list of lists; outer list is len(messages), inner list is len(sample_n)
    responses_set = await client.generate_multiple(
        messages=[
            [user_message("What is the capital of France? Return only the city name.")],
            [user_message("What is the capital of Italy? Return only the city name.")],
        ],
        sample_n=3,
    )
    total_duration = perf_counter() - start_time
    assert isinstance(responses_set, list)
    assert len(responses_set) == 2
    assert all(isinstance(responses, list) for responses in responses_set)
    assert all(len(responses) == 3 for responses in responses_set)
    for i, responses in enumerate(responses_set):
        for response in responses:
            assert isinstance(response, TextResponse)
            assert response.response
            if i == 0:
                assert 'Paris' in response.response
            else:
                assert 'Rome' in response.response
            assert response.total_tokens > 0
            assert response.total_cost > 0
            assert response.duration_seconds > 0
    assert total_duration > 0
    assert total_duration < sum(
        response.duration_seconds
        for responses in responses_set
        for response in responses
    ), \
        "Total duration should be less than the sum of individual durations because of concurrency"


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
