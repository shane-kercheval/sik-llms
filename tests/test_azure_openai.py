"""Tests for Azure OpenAI wrappers."""
from __future__ import annotations

from types import SimpleNamespace
import pytest

from sik_llms import (
    AzureOpenAI,
    AzureOpenAITools,
    Parameter,
    Tool,
    TextResponse,
    user_message,
)


class DummyResponses:
    """Stubbed responses client."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="Hello from Azure",
            output=[],
            usage=SimpleNamespace(input_tokens=5, output_tokens=7),
        )


class DummyAzureClient:
    """Stub AsyncAzureOpenAI client used for tests."""

    def __init__(self, *args, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        self.responses = DummyResponses()
        # Provide chat attribute so OpenAI base class expectations are met.
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=None, create=None))
        self.base_url = kwargs.get('azure_endpoint')


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://example-endpoint.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_DEPLOYMENT', 'test-deployment')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.setenv('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')
    yield
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_DEPLOYMENT', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_API_VERSION', raising=False)


@pytest.mark.asyncio
async def test_azure_openai_responses_stream(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('sik_llms.azure_openai.AsyncAzureOpenAI', DummyAzureClient)
    client = AzureOpenAI(model_name='gpt-4o-mini', use_responses_api=True)

    chunks = [response async for response in client.stream([user_message("ping")])]
    assert len(chunks) == 1
    response = chunks[0]
    assert isinstance(response, TextResponse)
    assert response.response == 'Hello from Azure'
    assert response.input_tokens == 5
    assert response.output_tokens == 7
    assert response.duration_seconds >= 0
    assert client.client.responses.calls[0]['model'] == 'test-deployment'


@pytest.mark.asyncio
async def test_azure_openai_delegates_to_openai_stream(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('sik_llms.azure_openai.AsyncAzureOpenAI', DummyAzureClient)

    async def fake_stream(self, messages):  # noqa: ANN001
        yield TextResponse(
            response='delegated',
            input_tokens=1,
            output_tokens=1,
            input_cost=None,
            output_cost=None,
            cache_read_tokens=None,
            cache_read_cost=None,
            duration_seconds=0.1,
        )

    monkeypatch.setattr('sik_llms.azure_openai.OpenAI.stream', fake_stream)
    client = AzureOpenAI(model_name='gpt-4o-mini')
    chunks = [response async for response in client.stream([user_message("ping")])]
    assert len(chunks) == 1
    assert chunks[0].response == 'delegated'


def test_azure_openai_tools_initialization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('sik_llms.azure_openai.AsyncAzureOpenAI', DummyAzureClient)
    tool = Tool(
        name='echo',
        description='Echo back text',
        parameters=[
            Parameter(
                name='text',
                param_type=str,
                required=True,
                description='Text to echo',
            ),
        ],
    )
    client = AzureOpenAITools(model_name='gpt-4o-mini', tools=[tool])
    assert client.deployment_name == 'test-deployment'
    assert client.model == 'test-deployment'
    assert client.model_parameters['tools'][0]['function']['name'] == 'echo'
