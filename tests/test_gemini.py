"""Tests for Gemini integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pydantic import BaseModel

from sik_llms import (
    StructuredOutputResponse,
    TextChunkEvent,
    TextResponse,
    create_client,
    user_message,
    Gemini,
    GeminiTools,
    Parameter,
    RegisteredClients,
    Tool,
    SUPPORTED_GEMINI_MODELS,
)
from sik_llms.gemini import num_tokens
from tests.conftest import GEMINI_TEST_MODEL, GEMINI_TEST_FUNCTION_CALLING
from types import SimpleNamespace
from google.genai import types

MonkeyPatch = pytest.MonkeyPatch


@pytest.fixture
def anyio_backend() -> str:  # pragma: no cover - test configuration helper
    return "asyncio"


@pytest.fixture(autouse=True)
def _patch_async_client(monkeypatch: MonkeyPatch) -> list[SimpleNamespace]:
    """Provide a dummy AsyncClient for Gemini tests."""
    instances = []

    class DummyModels(SimpleNamespace):
        pass

    class DummyAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.models = DummyModels()
            instances.append(self)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("sik_llms.gemini.genai.Client", DummyAsyncClient)
    return instances


@pytest.mark.usefixtures("_patch_async_client")
def test_create_client_returns_gemini() -> None:
    """create_client should instantiate a Gemini client when model is supported."""
    client = create_client(model_name=GEMINI_TEST_MODEL)
    assert isinstance(client, Gemini)
    assert client.model == SUPPORTED_GEMINI_MODELS[GEMINI_TEST_MODEL].model


@pytest.mark.usefixtures("_patch_async_client")
def test_client_instantiation_with_enum() -> None:
    """Client.instantiate should handle RegisteredClients.GEMINI."""
    client = create_client(
        model_name=GEMINI_TEST_MODEL,
        client_type=RegisteredClients.GEMINI,
    )
    assert isinstance(client, Gemini)


@pytest.mark.usefixtures("_patch_async_client")
def test_gemini_tools_configuration(monkeypatch: MonkeyPatch) -> None:
    """GeminiTools should configure tool declarations and tool config."""
    monkeypatch.setattr("sik_llms.gemini.genai.Client", lambda **_: SimpleNamespace())
    tool = Tool(
        name="add_numbers",
        description="Add two integers.",
        parameters=[
            Parameter(name="a", param_type=int, required=True),
            Parameter(name="b", param_type=int, required=True),
        ],
    )

    client = GeminiTools(
        model_name=GEMINI_TEST_FUNCTION_CALLING,
        tools=[tool],
    )

    assert client.tool_config.function_calling_config.mode == "ANY"
    declaration = client.declarations[0]
    assert declaration.name == "add_numbers"
    assert declaration.parameters.type == types.Type.OBJECT
    assert "a" in declaration.parameters.properties
    assert declaration.parameters.properties["a"].type == types.Type.INTEGER


@pytest.mark.anyio("asyncio")
@pytest.mark.usefixtures("_patch_async_client")
async def test_gemini_streaming() -> None:
    """Stream should yield chunk events followed by a TextResponse."""
    client = create_client(model_name=GEMINI_TEST_MODEL)

    class DummyStream:
        def __iter__(self):  # pragma: no cover - simple iterator
            yield SimpleNamespace(text="Hello")
            yield SimpleNamespace(text=" world")

        def get_final_response(self) -> SimpleNamespace:
            usage = SimpleNamespace(
                prompt_token_count=4,
                candidates_token_count=3,
                thoughts_token_count=0,
                cached_content_token_count=0,
            )
            return SimpleNamespace(usage_metadata=usage, text="Hello world")

    client.client.models.generate_content_stream = lambda **_kwargs: DummyStream()

    chunks = []
    final = None
    async for event in client.stream(messages=[user_message("Say hi")]):
        if isinstance(event, TextChunkEvent):
            chunks.append(event.content)
        else:
            final = event

    assert "".join(chunks) == "Hello world"
    assert isinstance(final, TextResponse)
    assert final.response == "Hello world"
    assert final.input_tokens == 4
    assert final.output_tokens == 3


class _SummaryModel(BaseModel):
    answer: str


@pytest.mark.usefixtures("_patch_async_client")
def test_gemini_structured_output(monkeypatch: MonkeyPatch) -> None:
    """Structured output responses should return parsed Pydantic models."""

    async def immediate_to_thread(
        func: Callable[..., Any],
        *args: object,
        **kwargs: object,
    ) -> object:  # pragma: no cover - helper
        return func(*args, **kwargs)

    monkeypatch.setattr("sik_llms.gemini.asyncio.to_thread", immediate_to_thread)

    client = create_client(model_name=GEMINI_TEST_MODEL, response_format=_SummaryModel)

    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )
    response = SimpleNamespace(
        parsed=_SummaryModel(answer="done"),
        usage_metadata=usage,
        candidates=[],
    )
    client.client.models.generate_content = lambda **_kwargs: response

    result = client(messages=[user_message("Return summary")])
    assert isinstance(result, StructuredOutputResponse)
    assert isinstance(result.parsed, _SummaryModel)
    assert result.parsed.answer == "done"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_gemini_num_tokens(monkeypatch: MonkeyPatch) -> None:
    """num_tokens should proxy through the Gemini SDK count call."""

    class DummyCountClient:
        def __init__(self, **_kwargs: object) -> None:
            self.models = SimpleNamespace(
                count_tokens=lambda *_args, **_kwargs: SimpleNamespace(total_tokens=42),
            )

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("sik_llms.gemini.genai.Client", DummyCountClient)

    tokens = num_tokens(GEMINI_TEST_MODEL, "hello")
    assert tokens == 42
