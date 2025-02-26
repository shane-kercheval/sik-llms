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
    ContentType,
    ReasoningEffort,
)
load_dotenv()

ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'
ANTRHOPIC_THINKING_MODEL = 'claude-3-7-sonnet-latest'


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


@pytest.mark.parametrize(
    "reasoning_effort",
    [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH],
)
@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
def test__Anthropic_instantiate__with_reasoning_effort(reasoning_effort: ReasoningEffort):
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
        reasoning_effort=reasoning_effort,
        max_tokens=4000,
    )
    assert isinstance(model, Anthropic)
    assert model.model_parameters.get('thinking') is not None
    assert model.model_parameters['thinking']['type'] == 'enabled'
    assert model.model_parameters['thinking']['budget_tokens']
    assert model.model_parameters['max_tokens'] > model.model_parameters['thinking']['budget_tokens']  # noqa: E501


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
def test__Anthropic_instantiate__with_thinking_budget():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
        thinking_budget_tokens=12000,
        max_tokens=4000,
    )
    assert isinstance(model, Anthropic)
    assert model.model_parameters.get('thinking') is not None
    assert model.model_parameters['thinking']['type'] == 'enabled'
    assert model.model_parameters['thinking']['budget_tokens'] == 12000
    assert model.model_parameters['max_tokens'] > model.model_parameters['thinking']['budget_tokens']  # noqa: E501


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__with_thinking__reasoning_effort():
    """Test that the extended thinking chunks have the correct content types."""
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTRHOPIC_THINKING_MODEL,
        reasoning_effort=ReasoningEffort.LOW,
    )
    # Use a prompt that should trigger some thinking content
    has_thinking_content = False
    has_text_content = False

    messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
    async for response in model.run_async(messages=messages):
        if isinstance(response, ChatChunkResponse):
            if response.content_type == ContentType.THINKING:
                has_thinking_content = True
            if response.content_type == ContentType.TEXT:
                has_text_content = True
        if isinstance(response, ChatResponseSummary):
            # Check that summary contains both thinking and answer
            assert "45" in response.content

    assert has_thinking_content, "No thinking content was generated"
    assert has_text_content, "No text content was generated"


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__with_thinking__thinking_budget_tokens():
    """Test that the extended thinking chunks have the correct content types."""
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTRHOPIC_THINKING_MODEL,
        thinking_budget_tokens=2000,
    )
    # Use a prompt that should trigger some thinking content
    has_thinking_content = False
    has_text_content = False
    has_summary = False

    messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
    async for response in model.run_async(messages=messages):
        if isinstance(response, ChatChunkResponse):
            if response.content_type == ContentType.THINKING:
                has_thinking_content = True
            if response.content_type == ContentType.TEXT:
                has_text_content = True
        if isinstance(response, ChatResponseSummary):
            # Check that summary contains both thinking and answer
            has_summary = True
            assert "45" in response.content

    assert has_thinking_content, "No thinking content was generated"
    assert has_text_content, "No text content was generated"
    assert has_summary, "No summary was generated"


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__with_thinking__temperature():
    """
    Test that the extended thinking works when temperature is set.

    From docs: "Thinking isn't compatible with temperature, top_p, or top_k modifications as well
    as forced tool use."

    Make sure it doesn't crash.
    """
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTRHOPIC_THINKING_MODEL,
        reasoning_effort=ReasoningEffort.LOW,
        temperature=0.5,
    )
    # Use a prompt that should trigger some thinking content
    has_thinking_content = False
    has_text_content = False

    messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
    async for response in model.run_async(messages=messages):
        if isinstance(response, ChatChunkResponse):
            if response.content_type == ContentType.THINKING:
                has_thinking_content = True
            if response.content_type == ContentType.TEXT:
                has_text_content = True
        if isinstance(response, ChatResponseSummary):
            # Check that summary contains both thinking and answer
            assert "45" in response.content

    assert has_thinking_content, "No thinking content was generated"
    assert has_text_content, "No text content was generated"



@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__with_thinking__test_redacted_thinking():
    """Test that the extended thinking chunks have the correct content types."""
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTRHOPIC_THINKING_MODEL,
        thinking_budget_tokens=2000,
    )
    # Use a prompt that should trigger some thinking content
    has_redacted_thinking_content = False
    has_thinking_content = False
    has_text_content = False
    has_summary = False

    # test redacted thinking:
    # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks
    messages = [user_message("ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB")]  # noqa: E501
    async for response in model.run_async(messages=messages):
        if isinstance(response, ChatChunkResponse):
            if response.content_type == ContentType.REDACTED_THINKING:
                has_redacted_thinking_content = True
                assert response.content
            if response.content_type == ContentType.THINKING:
                has_thinking_content = True
            if response.content_type == ContentType.TEXT:
                has_text_content = True
        if isinstance(response, ChatResponseSummary):
            has_summary = True
            assert response.content

    assert has_redacted_thinking_content, "No redacted thinking content was generated"
    assert not has_thinking_content, "Thinking content was generated"
    assert has_text_content, "No text content was generated"
    assert has_summary, "No summary was generated"


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__without_thinking__verify_no_thinking_events():
    """Test that the extended thinking chunks have the correct content types."""
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTRHOPIC_THINKING_MODEL,
    )
    # Use a prompt that should trigger some thinking content
    has_thinking_content = False
    has_text_content = False
    has_summary = False

    messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
    async for response in model.run_async(messages=messages):
        if isinstance(response, ChatChunkResponse):
            if response.content_type == ContentType.THINKING:
                has_thinking_content = True
            if response.content_type == ContentType.TEXT:
                has_text_content = True
        if isinstance(response, ChatResponseSummary):
            # Check that summary contains both thinking and answer
            has_summary = True

    assert not has_thinking_content, "Thinking content was generated"
    assert has_text_content, "No text content was generated"
    assert has_summary, "No summary was generated"
