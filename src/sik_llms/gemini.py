"""Helper functions for interacting with the Google Gemini API."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from copy import deepcopy
from datetime import date
from time import perf_counter
from collections.abc import AsyncGenerator
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from sik_llms.models_base import (
    Client,
    ImageContent,
    ImageSourceType,
    ModelInfo,
    ModelProvider,
    RegisteredClients,
    StructuredOutputResponse,
    TextChunkEvent,
    TextResponse,
    Tool,
    ToolChoice,
    ToolPrediction,
    ToolPredictionResponse,
)
from sik_llms.utilities import get_json_schema_type


load_dotenv()


# Pricing is stored per-token (per 1M tokens on the pricing page).
GEMINI_MODEL_INFOS = [
    ModelInfo(
        model="gemini-2.5-pro",
        provider=ModelProvider.GOOGLE,
        max_output_tokens=65_536,
        context_window_size=1_048_576,
        pricing={
            "input": 1.25 / 1_000_000,
            "output": 10.00 / 1_000_000,
            "input_long": 2.50 / 1_000_000,
            "output_long": 15.00 / 1_000_000,
            "cached": 0.31 / 1_000_000,
        },
        supports_reasoning=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2025, month=1, day=1),
    ),
    ModelInfo(
        model="gemini-2.5-flash",
        provider=ModelProvider.GOOGLE,
        max_output_tokens=65_536,
        context_window_size=1_048_576,
        pricing={
            "input": 0.30 / 1_000_000,
            "output": 2.50 / 1_000_000,
            "cached": 0.075 / 1_000_000,
        },
        supports_reasoning=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2025, month=1, day=1),
    ),
    ModelInfo(
        model="gemini-2.5-flash-lite",
        provider=ModelProvider.GOOGLE,
        max_output_tokens=65_536,
        context_window_size=1_048_576,
        pricing={
            "input": 0.10 / 1_000_000,
            "output": 0.40 / 1_000_000,
            "cached": 0.025 / 1_000_000,
        },
        supports_reasoning=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2025, month=1, day=1),
    ),
]


SUPPORTED_GEMINI_MODELS = {model.model: model for model in GEMINI_MODEL_INFOS}


def _ensure_api_key(api_key: str | None) -> str:
    """Return a Gemini API key, preferring explicit input then environment variables."""
    env_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not env_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set.")
    return env_key


def _convert_to_parts(content: str | list[str | ImageContent]) -> list[types.Part]:
    """Convert internal message content into Gemini parts."""
    parts: list[types.Part] = []
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            parts.append(types.Part.from_text(text=stripped))
        return parts or [types.Part.from_text(text="")]

    if not isinstance(content, list):
        raise TypeError(f"Unsupported message content type: {type(content)}")

    for item in content:
        if isinstance(item, str):
            stripped = item.strip()
            parts.append(types.Part.from_text(text=stripped))
        elif isinstance(item, ImageContent):
            if item.source_type == ImageSourceType.URL:
                parts.append(
                    types.Part.from_uri(
                        uri=item.data,
                        mime_type=item.media_type or "image/png",
                    ),
                )
            elif item.source_type == ImageSourceType.BASE64:
                binary = base64.b64decode(item.data)
                parts.append(
                    types.Part.from_bytes(
                        data=binary,
                        mime_type=item.media_type or "image/png",
                    ),
                )
        else:
            raise TypeError(f"Unsupported content item type: {type(item)}")

    return parts or [types.Part.from_text(text="")]


def _convert_messages(
    messages: list[dict[str, Any]],
    cache_content: list[str] | str | None,
) -> tuple[str | None, list[types.Content]]:
    """Convert internal messages into Gemini system instruction and content list."""
    system_segments: list[str] = []
    contents: list[types.Content] = []

    if cache_content:
        cached_list = [cache_content] if isinstance(cache_content, str) else list(cache_content)
        for cached in cached_list:
            contents.append(types.Content(role="user", parts=_convert_to_parts(cached)))

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_segments.append(content.strip())
            elif isinstance(content, list):
                system_segments.extend(
                    segment.strip() for segment in content if isinstance(segment, str)
                )
            else:
                raise TypeError(f"Unsupported system content type: {type(content)}")
            continue

        if role not in {"user", "assistant", "model"}:
            raise ValueError(f"Unsupported role `{role}` for Gemini messages.")

        gemini_role = "user" if role == "user" else "model"
        parts = _convert_to_parts(content)
        contents.append(types.Content(role=gemini_role, parts=parts))

    system_instruction = "\n\n".join(segment for segment in system_segments if segment) or None
    return system_instruction, contents


def _create_generation_config(
    model_parameters: dict[str, Any],
    response_format: type[BaseModel] | None,
) -> types.GenerateContentConfig | None:
    """Create the Gemini generation config for the request."""
    if not model_parameters and not response_format:
        return None

    config_kwargs = deepcopy(model_parameters) if model_parameters else {}
    if response_format:
        config_kwargs.setdefault("response_mime_type", "application/json")
        config_kwargs["response_schema"] = response_format

    return types.GenerateContentConfig(**config_kwargs)


def _collect_usage_metadata(response) -> tuple[int, int, int | None]:  # noqa: ANN001
    """Return token accounting from a Gemini response."""
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return 0, 0, 0, None

    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    cached_tokens = getattr(usage, "cached_content_token_count", 0)
    total_output_tokens = output_tokens + thoughts_tokens
    return input_tokens, total_output_tokens, cached_tokens


def _calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int | None,
    pricing_lookup: dict[str, float] | None,
) -> tuple[float | None, float | None, float | None]:
    """Calculate token-based costs using the model pricing metadata."""
    if not pricing_lookup:
        return None, None, None

    input_cost = input_tokens * pricing_lookup.get("input", 0)
    output_cost = output_tokens * pricing_lookup.get("output", 0)
    cache_cost = None
    if cached_tokens:
        cache_rate = pricing_lookup.get("cached") or pricing_lookup.get("cache", 0)
        cache_cost = cached_tokens * cache_rate if cache_rate else None
    return input_cost, output_cost, cache_cost


def _model_parameters_without_tools(model_parameters: dict[str, Any]) -> dict[str, Any]:
    """Return model parameters without tool-specific keys."""
    return {
        key: value
        for key, value in model_parameters.items()
        if key not in {"tools", "tool_config"}
    }


def _tool_to_schema(tool: Tool) -> dict[str, Any]:
    properties = {}
    required: list[str] = []

    for param in tool.parameters:
        try:
            json_type, extra = get_json_schema_type(param.param_type)
        except ValueError:
            json_type, extra = "string", {}

        schema: dict[str, Any] = {"type": json_type, **extra}
        if param.description:
            schema["description"] = param.description
        if param.valid_values:
            schema["enum"] = param.valid_values
        if param.any_of:
            any_of = []
            for union_type in param.any_of:
                try:
                    union_json_type, union_extra = get_json_schema_type(union_type)
                except ValueError:
                    union_json_type, union_extra = "string", {}
                any_of.append({"type": union_json_type, **union_extra})
            schema["anyOf"] = any_of
        properties[param.name] = schema
        if param.required:
            required.append(param.name)

    schema_dict: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema_dict["required"] = required
    return schema_dict


def _tools_to_gemini_tools(
    tools: list[Tool],
) -> tuple[list[types.Tool], list[types.FunctionDeclaration]]:
    declarations: list[types.FunctionDeclaration] = []
    for tool in tools:
        declarations.append(
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=_tool_to_schema(tool),
            ),
        )
    gemini_tool = types.Tool(function_declarations=declarations)
    return [gemini_tool], declarations


def _tool_response_to_prediction(response) -> tuple[ToolPrediction | None, str | None]:  # noqa: ANN001
    """Extract tool prediction or assistant message from a Gemini response."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None, None

    candidate = candidates[0]
    parts = getattr(candidate, "content", None)
    if parts is None:
        return None, None

    for part in parts.parts:
        function_call = getattr(part, "function_call", None)
        if function_call:
            arguments = function_call.args
            if hasattr(arguments, "to_dict"):
                arguments = arguments.to_dict()
            elif hasattr(arguments, "items"):
                arguments = dict(arguments)
            if arguments is not None and not isinstance(arguments, dict | list):
                arguments = json.loads(json.dumps(arguments))
            return (
                ToolPrediction(
                    name=function_call.name,
                    arguments=arguments or {},
                    call_id=function_call.name,
                ),
                None,
            )

        text = getattr(part, "text", None)
        if text:
            return None, text

    return None, None


@Client.register(RegisteredClients.GEMINI)
class Gemini(Client):
    """Wrapper around the Gemini SDK for text and multimodal responses."""

    def __init__(
        self,
        model_name: str,
        response_format: type[BaseModel] | None = None,
        cache_content: list[str] | str | None = None,
        api_key: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        **model_kwargs: object,
    ) -> None:
        model_info = SUPPORTED_GEMINI_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' is not supported for Gemini.")

        api_key_value = _ensure_api_key(api_key)
        client_kwargs = client_kwargs or {}
        client_kwargs.setdefault("api_key", api_key_value)
        self.client = genai.Client(**client_kwargs)
        self.model = model_info.model
        self.response_format = response_format
        self.cache_content = cache_content
        self.model_parameters = {
            key: value for key, value in model_kwargs.items() if value is not None
        }

    async def stream(  # noqa: PLR0915
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[TextChunkEvent | TextResponse | StructuredOutputResponse, None]:
        """Stream Gemini responses, yielding text chunks and final summaries."""
        model_info = SUPPORTED_GEMINI_MODELS.get(self.model)
        pricing_lookup = model_info.pricing if model_info else None

        system_instruction, contents = _convert_messages(messages, self.cache_content)
        config = _create_generation_config(self.model_parameters, self.response_format)

        request_kwargs = {
            "model": self.model,
            "contents": contents,
        }
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction
        if config:
            request_kwargs["config"] = config

        if self.response_format:
            start_time = perf_counter()
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                **request_kwargs,
            )
            end_time = perf_counter()
            input_tokens, output_tokens, cached_tokens = _collect_usage_metadata(response)
            input_cost, output_cost, cache_cost = _calculate_cost(
                input_tokens,
                output_tokens,
                cached_tokens,
                pricing_lookup,
            )

            parsed = getattr(response, "parsed", None)
            if parsed is None and response.candidates:
                candidate = response.candidates[0]
                parsed = next(
                    (
                        json.loads(part.text)
                        for part in candidate.content.parts
                        if getattr(part, "text", None)
                    ),
                    None,
                )
                if parsed and self.response_format:
                    parsed = self.response_format.model_validate(parsed)

            refusal = None
            if (
                response.candidates
                and getattr(response.candidates[0], "finish_reason", None) == "SAFETY"
            ):
                refusal = "Response blocked for safety reasons."

            yield StructuredOutputResponse(
                parsed=parsed,
                refusal=refusal,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cached_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                cache_read_cost=cache_cost,
                duration_seconds=end_time - start_time,
            )
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        final_future = loop.create_future()

        def worker() -> None:
            try:
                stream = self.client.models.generate_content_stream(**request_kwargs)
                for chunk in stream:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                final_response = stream.get_final_response()
                if not final_future.done():
                    loop.call_soon_threadsafe(final_future.set_result, final_response)
            except Exception as exc:
                if not final_future.done():
                    loop.call_soon_threadsafe(final_future.set_exception, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        chunks: list[str] = []
        start_time = perf_counter()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            text = getattr(chunk, "text", None)
            if text:
                chunks.append(text)
                yield TextChunkEvent(content=text)

        response = await final_future
        end_time = perf_counter()
        input_tokens, output_tokens, cached_tokens = _collect_usage_metadata(response)
        input_cost, output_cost, cache_cost = _calculate_cost(
            input_tokens,
            output_tokens,
            cached_tokens,
            pricing_lookup,
        )

        final_text = "".join(chunks) or getattr(response, "text", "")
        yield TextResponse(
            response=final_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_cost,
            duration_seconds=end_time - start_time,
        )


@Client.register(RegisteredClients.GEMINI_TOOLS)
class GeminiTools(Client):
    """Wrapper around Gemini function calling support."""

    def __init__(
        self,
        model_name: str,
        tools: list[Tool],
        tool_choice: ToolChoice = ToolChoice.REQUIRED,
        api_key: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        **model_kwargs: object,
    ) -> None:
        if not tools:
            raise ValueError("At least one tool must be provided for Gemini tools client.")
        model_info = SUPPORTED_GEMINI_MODELS.get(model_name)
        if not model_info or not model_info.supports_tools:
            raise ValueError(f"Model '{model_name}' is not supported for Gemini tools.")

        api_key_value = _ensure_api_key(api_key)
        client_kwargs = client_kwargs or {}
        client_kwargs.setdefault("api_key", api_key_value)
        self.client = genai.Client(**client_kwargs)
        self.model = model_info.model
        self.model_parameters = {
            key: value for key, value in model_kwargs.items() if value is not None
        }
        self.tools, self.declarations = _tools_to_gemini_tools(tools)
        mode = "ANY" if tool_choice == ToolChoice.REQUIRED else "AUTO"
        self.tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=mode),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[ToolPredictionResponse, None]:
        """Return Gemini tool predictions for a conversation."""
        model_info = SUPPORTED_GEMINI_MODELS.get(self.model)
        pricing_lookup = model_info.pricing if model_info else None

        system_instruction, contents = _convert_messages(messages, cache_content=None)
        parameters = _model_parameters_without_tools(self.model_parameters)
        config = types.GenerateContentConfig(**parameters) if parameters else None
        if config:
            config.tools = self.tools
            config.tool_config = self.tool_config
        else:
            config = types.GenerateContentConfig(
                tools=self.tools,
                tool_config=self.tool_config,
            )

        request_kwargs = {
            "model": self.model,
            "contents": contents,
            "config": config,
        }
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction

        start_time = perf_counter()
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            **request_kwargs,
        )
        end_time = perf_counter()
        input_tokens, output_tokens, cached_tokens = _collect_usage_metadata(response)
        input_cost, output_cost, cache_cost = _calculate_cost(
            input_tokens,
            output_tokens,
            cached_tokens,
            pricing_lookup,
        )

        tool_prediction, message = _tool_response_to_prediction(response)

        yield ToolPredictionResponse(
            tool_prediction=tool_prediction,
            message=message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_cost,
            duration_seconds=end_time - start_time,
        )


def num_tokens(model_name: str, content: str) -> int:
    """Return the token count for a single string using the Gemini tokenizer."""
    api_key_value = _ensure_api_key(None)
    client = genai.Client(api_key=api_key_value)
    response = client.models.count_tokens(model=model_name, contents=content)
    return getattr(response, "total_tokens", 0)


def num_tokens_from_messages(model_name: str, messages: list[dict[str, Any]]) -> int:
    """Return the token count for a message history."""
    api_key_value = _ensure_api_key(None)
    client = genai.Client(api_key=api_key_value)
    _, contents = _convert_messages(messages, cache_content=None)
    response = client.models.count_tokens(model=model_name, contents=contents)
    return getattr(response, "total_tokens", 0)
