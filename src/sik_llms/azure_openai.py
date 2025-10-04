"""Azure OpenAI client wrappers."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from copy import deepcopy
import os
from time import perf_counter
from typing import Any, Callable

from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from sik_llms.models_base import (
    Client,
    ModelInfo,
    ModelProvider,
    RegisteredClients,
    ReasoningEffort,
    Tool,
    ToolChoice,
    TextChunkEvent,
    StructuredOutputResponse,
    TextResponse,
)
from sik_llms.openai import (
    OpenAI,
    OpenAITools,
    SUPPORTED_OPENAI_MODELS,
    _convert_messages,
)


DEFAULT_AZURE_API_VERSION = '2025-04-01-preview'


def _convert_model_info(info: ModelInfo) -> ModelInfo:
    """Create an Azure-specific copy of ModelInfo."""
    azure_info = info.model_copy()
    azure_info.provider = ModelProvider.AZURE_OPENAI
    # Azure pricing varies per region and is not aligned 1:1 with OpenAI list pricing.
    azure_info.pricing = None
    metadata = azure_info.metadata.copy() if azure_info.metadata else {}
    metadata['requires_deployment'] = True
    metadata['default_api_version'] = DEFAULT_AZURE_API_VERSION
    azure_info.metadata = metadata
    return azure_info


SUPPORTED_AZURE_OPENAI_MODELS: dict[str, ModelInfo] = {
    name: _convert_model_info(model)
    for name, model in SUPPORTED_OPENAI_MODELS.items()
}


def _resolve_endpoint(endpoint: str | None) -> str:
    endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint must be provided via argument or AZURE_OPENAI_ENDPOINT")
    return endpoint.rstrip('/')


def _resolve_deployment(model_name: str | None, deployment_name: str | None) -> str:
    deployment = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT') or model_name
    if not deployment:
        raise ValueError("Azure OpenAI deployment name must be provided via argument or AZURE_OPENAI_DEPLOYMENT")
    return deployment


def _resolve_api_version(api_version: str | None) -> str:
    return api_version or os.getenv('AZURE_OPENAI_API_VERSION') or DEFAULT_AZURE_API_VERSION


def _resolve_credentials(
    api_key: str | None,
    azure_ad_token: str | None,
    azure_ad_token_provider: Callable[[], str] | None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
    azure_ad_token = azure_ad_token or os.getenv('AZURE_OPENAI_AD_TOKEN')
    if not api_key and not azure_ad_token and not azure_ad_token_provider:
        raise ValueError("Azure OpenAI requires either an API key or an Azure AD token provider")
    credentials: dict[str, Any] = {}
    if api_key:
        credentials['api_key'] = api_key
    if azure_ad_token:
        credentials['azure_ad_token'] = azure_ad_token
    if azure_ad_token_provider:
        credentials['azure_ad_token_provider'] = azure_ad_token_provider
    return credentials


@Client.register(RegisteredClients.AZURE_OPENAI)
class AzureOpenAI(OpenAI):
    """Azure OpenAI chat wrapper with optional Responses API support."""

    def __init__(
            self,
            model_name: str | None = None,
            *,
            deployment_name: str | None = None,
            azure_endpoint: str | None = None,
            api_version: str | None = None,
            api_key: str | None = None,
            azure_ad_token: str | None = None,
            azure_ad_token_provider: Callable[[], str] | None = None,
            default_headers: dict[str, str] | None = None,
            default_query: dict[str, str] | None = None,
            reasoning_effort: ReasoningEffort | None = None,
            response_format: type[BaseModel] | None = None,
            cache_content: list[str] | str | None = None,
            pricing_override: dict[str, float] | None = None,
            use_responses_api: bool = False,
            responses_kwargs: dict[str, Any] | None = None,
            **model_kwargs: dict,
        ) -> None:
        self.azure_endpoint = _resolve_endpoint(azure_endpoint)
        self.api_version = _resolve_api_version(api_version)
        deployment_identifier = _resolve_deployment(model_name, deployment_name)
        credentials = _resolve_credentials(api_key, azure_ad_token, azure_ad_token_provider)

        self.use_responses_api = use_responses_api
        if use_responses_api and response_format is not None:
            raise ValueError("Structured output parsing is not yet supported when use_responses_api=True")
        self.responses_kwargs = responses_kwargs or {}

        client = AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            default_headers=default_headers,
            default_query=default_query,
            **credentials,
        )

        model_lookup_name = model_name or deployment_identifier
        # Azure deployments always require the deployment name for API calls.
        super().__init__(
            model_name=model_lookup_name,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            cache_content=cache_content,
            server_url=self.azure_endpoint,
            client=client,
            supported_models=SUPPORTED_AZURE_OPENAI_MODELS,
            api_model_name=deployment_identifier,
            pricing_override=pricing_override,
            **model_kwargs,
        )
        self.deployment_name = deployment_identifier

    async def stream(
            self,
            messages: list[dict[str, object]],
        ) -> AsyncGenerator[TextChunkEvent | TextResponse | StructuredOutputResponse, None]:
        if not self.use_responses_api:
            async for chunk in super().stream(messages):
                yield chunk
            return

        # Responses API call (non-streaming) to provide a TextResponse summary.
        converted_messages = _convert_messages(messages, self.cache_content)
        request_parameters = deepcopy(self.model_parameters)
        request_parameters.update(self.responses_kwargs)
        start_time = perf_counter()
        completion = await self.client.responses.create(
            model=self.model,
            input=converted_messages,
            **request_parameters,
        )
        end_time = perf_counter()

        text_output = getattr(completion, 'output_text', None) or ''
        if not text_output and getattr(completion, 'output', None):
            for item in completion.output:
                for content in getattr(item, 'content', []) or []:
                    if getattr(content, 'type', '') == 'output_text':
                        text_output += getattr(content, 'text', '')

        usage = getattr(completion, 'usage', None)
        input_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0

        pricing_lookup = self.pricing_lookup
        input_cost = (
            input_tokens * pricing_lookup['input']
            if pricing_lookup and 'input' in pricing_lookup
            else None
        )
        output_cost = (
            output_tokens * pricing_lookup['output']
            if pricing_lookup and 'output' in pricing_lookup
            else None
        )

        yield TextResponse(
            response=text_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_tokens=None,
            cache_read_cost=None,
            duration_seconds=end_time - start_time,
        )


@Client.register(RegisteredClients.AZURE_OPENAI_TOOLS)
class AzureOpenAITools(OpenAITools):
    """Azure OpenAI tools/function calling wrapper."""

    def __init__(
            self,
            model_name: str | None = None,
            tools: list[Tool] | None = None,
            *,
            deployment_name: str | None = None,
            azure_endpoint: str | None = None,
            api_version: str | None = None,
            api_key: str | None = None,
            azure_ad_token: str | None = None,
            azure_ad_token_provider: Callable[[], str] | None = None,
            default_headers: dict[str, str] | None = None,
            default_query: dict[str, str] | None = None,
            tool_choice: ToolChoice = ToolChoice.REQUIRED,
            pricing_override: dict[str, float] | None = None,
            **model_kwargs: dict,
        ) -> None:
        if tools is None:
            tools = []
        azure_endpoint_resolved = _resolve_endpoint(azure_endpoint)
        api_version_resolved = _resolve_api_version(api_version)
        deployment_identifier = _resolve_deployment(model_name, deployment_name)
        credentials = _resolve_credentials(api_key, azure_ad_token, azure_ad_token_provider)

        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint_resolved,
            api_version=api_version_resolved,
            default_headers=default_headers,
            default_query=default_query,
            **credentials,
        )

        model_lookup_name = model_name or deployment_identifier
        super().__init__(
            model_name=model_lookup_name,
            tools=tools,
            tool_choice=tool_choice,
            server_url=azure_endpoint_resolved,
            client=client,
            supported_models=SUPPORTED_AZURE_OPENAI_MODELS,
            api_model_name=deployment_identifier,
            pricing_override=pricing_override,
            **model_kwargs,
        )
        self.azure_endpoint = azure_endpoint_resolved
        self.api_version = api_version_resolved
        self.deployment_name = deployment_identifier
