"""Helper functions for OpenAI API."""
from copy import deepcopy
from datetime import date
from functools import cache
import json
import os
from time import perf_counter
from collections.abc import AsyncGenerator
from openai import AsyncOpenAI
from pydantic import BaseModel
import tiktoken
from tiktoken import Encoding
from sik_llms.models_base import (
    ImageContent,
    ImageSourceType,
    ModelInfo,
    ModelProvider,
    Tool,
    ToolPredictionResponse,
    ToolPrediction,
    Client,
    TextChunkEvent,
    TextResponse,
    ReasoningEffort,
    RegisteredClients,
    StructuredOutputResponse,
    ToolChoice,
)
from dotenv import load_dotenv

from sik_llms.utilities import _remove_defaults_recursively, get_json_schema_type
from sik_llms.telemetry import safe_span
load_dotenv()

# Define all OpenAI models
OPENAI_MODEL_INFOS = [
    ####
    # GPT-4.1
    ####
    ModelInfo(
        model='gpt-4.1-2025-04-14',
        provider=ModelProvider.OPENAI,
        max_output_tokens=32_768,
        context_window_size=1_047_576,
        pricing={
            'input': 2.00 / 1_000_000, 'output': 8.00 / 1_000_000,
            'cached': 0.50 / 1_000_000,
        },
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2024, month=5, day=31),
    ),
    ModelInfo(
        model='gpt-4.1-mini-2025-04-14',
        provider=ModelProvider.OPENAI,
        max_output_tokens=32_768,
        context_window_size=1_047_576,
        pricing={
            'input': 0.40 / 1_000_000, 'output': 1.60 / 1_000_000,
            'cached': 0.10 / 1_000_000,
        },
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2024, month=5, day=31),
    ),
    ModelInfo(
        model='gpt-4.1-nano-2025-04-14',
        provider=ModelProvider.OPENAI,
        max_output_tokens=32_768,
        context_window_size=1_047_576,
        pricing={
            'input': 0.10 / 1_000_000, 'output': 0.40 / 1_000_000,
            'cached': 0.025 / 1_000_000,
        },
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2024, month=5, day=31),
    ),
    ####
    # GPT-4o
    ####
    ModelInfo(
        model='gpt-4o-mini-2024-07-18',
        provider=ModelProvider.OPENAI,
        max_output_tokens=16_384,
        context_window_size=128_000,
        pricing={
            'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000,
            'cached': 0.075 / 1_000_000,
        },
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2023, month=9, day=30),
    ),

    ModelInfo(
        model='gpt-4o-2024-08-06',
        provider=ModelProvider.OPENAI,
        max_output_tokens=16_384,
        context_window_size=128_000,
        pricing={
            'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000,
            'cached': 1.25 / 1_000_000,
        },
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2023, month=9, day=30),
    ),
    ModelInfo(
        model='gpt-4o-2024-11-20',
        provider=ModelProvider.OPENAI,
        max_output_tokens=16_384,
        context_window_size=128_000,
        pricing={
            'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000,
            'cached': 1.25 / 1_000_000,
        },
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2023, month=9, day=30),
    ),
    ####
    # o3
    ####
    ModelInfo(
        model='o3-mini-2025-01-31',
        provider=ModelProvider.OPENAI,
        max_output_tokens=100_000,
        context_window_size=200_000,
        pricing={
            'input': 1.10 / 1_000_000, 'output': 4.40 / 1_000_000,
            'cached': 0.55 / 1_000_000,
        },
        supports_reasoning=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2023, month=9, day=30),
    ),
    ####
    # o1
    ####
    ModelInfo(
        model='o1-2024-12-17',
        provider=ModelProvider.OPENAI,
        max_output_tokens=100_000,
        context_window_size=200_000,
        pricing={
            'input': 15.00 / 1_000_000, 'output': 60.00 / 1_000_000,
            'cached': 7.50 / 1_000_000,
        },
        supports_reasoning=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2023, month=9, day=30),
    ),
]

SUPPORTED_OPENAI_MODELS = {model.model: model for model in OPENAI_MODEL_INFOS}
SUPPORTED_OPENAI_MODELS['gpt-4.1'] = SUPPORTED_OPENAI_MODELS['gpt-4.1-2025-04-14']
SUPPORTED_OPENAI_MODELS['gpt-4.1-mini'] = SUPPORTED_OPENAI_MODELS['gpt-4.1-mini-2025-04-14']
SUPPORTED_OPENAI_MODELS['gpt-4.1-nano'] = SUPPORTED_OPENAI_MODELS['gpt-4.1-nano-2025-04-14']
SUPPORTED_OPENAI_MODELS['gpt-4o'] = SUPPORTED_OPENAI_MODELS['gpt-4o-2024-11-20']
SUPPORTED_OPENAI_MODELS['gpt-4o-mini'] = SUPPORTED_OPENAI_MODELS['gpt-4o-mini-2024-07-18']
SUPPORTED_OPENAI_MODELS['o3-mini'] = SUPPORTED_OPENAI_MODELS['o3-mini-2025-01-31']
SUPPORTED_OPENAI_MODELS['o1'] = SUPPORTED_OPENAI_MODELS['o1-2024-12-17']


@cache
def _get_encoding_for_model(model_name: str) -> Encoding:
    """Gets the encoding for a given model so that we can calculate the number of tokens."""
    return tiktoken.encoding_for_model(model_name)


def num_tokens(model_name: str, value: str) -> int:
    """For a given model, returns the number of tokens based on the str `value`."""
    return len(_get_encoding_for_model(model_name=model_name).encode(value))


def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """
    Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Returns the number of tokens used by a list of messages.
    """
    try:
        encoding = _get_encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    if model_name in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model_name:
        return num_tokens_from_messages(messages=messages, model_name="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model_name:
        return num_tokens_from_messages(messages=messages, model_name="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model_name:
        return num_tokens_from_messages(messages=messages, model_name="gpt-4o-2024-08-06")
    elif "gpt-4" in model_name:
        return num_tokens_from_messages(messages=messages, model_name="gpt-4-0613")
    elif "o1" in model_name or "o3-mini" in model_name:
        return num_tokens_from_messages(messages=messages, model_name="gpt-4o-2024-08-06")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model_name}.""",
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _parse_completion_chunk(chunk) -> TextChunkEvent:  # noqa: ANN001
    if chunk.object != 'chat.completion.chunk':
        raise ValueError(f"Unexpected object type: {chunk.object}")
    log_prob = None
    if chunk.choices[0].logprobs:
        log_prob = chunk.choices[0].logprobs.content[0].logprob
    return TextChunkEvent(
        content=chunk.choices[0].delta.content,
        logprob=log_prob,
    )


def _convert_messages(
        messages: list[dict[str, str | list[str | ImageContent]]],
        cache_content: list[str] | str | None = None,
    ) -> list[dict[str, str | list[dict]]]:
    """Converts messages to the format expected by the OpenAI API."""
    messages = deepcopy(messages)

    # Handle cache content first
    if cache_content:
        if isinstance(cache_content, str):
            cache_content = [cache_content]
        # find index after last system message but before the first user message
        for i, message in enumerate(messages):
            if message.get('role') == 'user':
                break
        else:
            i = len(messages)

        cached_messages = [
            {'role': 'user', 'content': content.strip()}
            for content in cache_content
        ]
        messages[i:i] = cached_messages

    # Handle image content
    converted_messages = []
    for message in messages:
        if isinstance(message['content'], list):
            converted_content = []
            for item in message['content']:
                if isinstance(item, str):
                    converted_content.append({
                        "type": "text",
                        "text": item.strip(),
                    })
                elif isinstance(item, ImageContent):
                    if item.source_type == ImageSourceType.URL:
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {"url": item.data},
                        })
                    elif item.source_type == ImageSourceType.BASE64:
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.media_type};base64,{item.data}",
                            },
                        })
            message = {**message, 'content': converted_content}  # noqa: PLW2901
        converted_messages.append(message)
    return converted_messages


@Client.register(RegisteredClients.OPENAI)
class OpenAI(Client):
    """
    Wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.

    The user can specify the model name, timeout, stream, and other parameters for the API call
    either in the constructor or when calling the object. If the latter, the parameters specified
    when calling the object will override the parameters specified in the constructor.
    """

    def __init__(
            self,
            model_name: str | None = None,
            reasoning_effort: ReasoningEffort | None = None,
            response_format: type[BaseModel] | None = None,
            cache_content: list[str] | str | None = None,
            server_url: str | None = None,
            api_key: str | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        NOTE: max_tokens is deprecated in favor of max_completion_tokens and, if used, will be
        renamed to max_completion_tokens.

        If not provided, `max_completion_tokens` will be set to 8_000 for OpenAI models (i.e.
        we do not set it if the openai api is being used for a different provider/models).

        Args:
            client:
                An instance of the AsyncOpenAI client.
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            reasoning_effort:
                The reasoning effort to use for the API call (reasoning model required).
            response_format:
                Pydantic class defining structure of the response (i.e. structured output)
            cache_content:
                List of strings to cache for the model. These strings are inserted as the first
                user message.
            server_url:
                The base URL for the API call.
            api_key:
                The API key to use for the API call. If not provided, the API key will be read from
                the OPENAI_API_KEY environment variable.
            **model_kwargs: Additional parameters to pass to the API call
        """
        super().__init__(model_name, **model_kwargs)

        model_info = SUPPORTED_OPENAI_MODELS.get(model_name)
        if server_url is None and not model_info:
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.server_url = server_url
        self.client = AsyncOpenAI(
            base_url=server_url,
            api_key=api_key or os.getenv('OPENAI_API_KEY') or 'None',
        )
        self.model = model_name
        self.model_parameters = model_kwargs or {}
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self.cache_content = cache_content
        if response_format:
            self.model_parameters['response_format'] = response_format
        if reasoning_effort:
            if isinstance(reasoning_effort, ReasoningEffort):
                self.model_parameters['reasoning_effort'] = reasoning_effort.value
            else:
                self.model_parameters['reasoning_effort'] = reasoning_effort
        # logprobs, temp, top_p are not supported with reasoning or reasoning models
        if reasoning_effort or (self.model and any(self.model.startswith(prefix) for prefix in ['o1', 'o3'])):  # noqa: E501
            self.model_parameters.pop('logprobs', None)
            self.model_parameters.pop('temperature', None)
            self.model_parameters.pop('top_p', None)

        # NOTE: max_tokens is deprecated in favor of max_completion_tokens
        # NOTE that they do have different meanings so popping is not technically correct; but yolo
        max_tokens = self.model_parameters.pop('max_tokens', None)
        if max_tokens:
            self.model_parameters['max_completion_tokens'] = max_tokens
        # if we're using an openai model, let's set default max_completion_tokens
        if 'max_completion_tokens' not in self.model_parameters and model_info:
            self.model_parameters['max_completion_tokens'] = min(8_000, model_info.max_output_tokens)  # noqa: E501

    async def stream(  # noqa: PLR0912, PLR0915
            self,
            messages: list[dict],
        ) -> AsyncGenerator[TextChunkEvent | TextResponse | None]:
        """
        Streams chat chunks and returns a final summary. Note that any parameters passed to this
        method will override the parameters passed to the constructor.
        """
        with safe_span(
            self.tracer,
            "llm.openai.stream",
            attributes={
                "llm.model": self.model,
                "llm.provider": "openai",
                "llm.streaming": True,
            },
        ) as span:
            # Add OpenAI-specific attributes
            if span:
                span.set_attribute("llm.api.vendor", "openai")
                if hasattr(self, 'temperature') and 'temperature' in self.model_parameters:
                    span.set_attribute("llm.request.temperature", self.model_parameters['temperature'])  # noqa: E501
                if hasattr(self, 'max_completion_tokens') and 'max_completion_tokens' in self.model_parameters:  # noqa: E501
                    span.set_attribute("llm.request.max_tokens", self.model_parameters['max_completion_tokens'])  # noqa: E501

            model_info = SUPPORTED_OPENAI_MODELS.get(self.model)
            pricing_lookup = model_info.pricing if model_info else None

            try:
                messages = _convert_messages(messages, self.cache_content)
                model_parameters = deepcopy(self.model_parameters)
                if self.response_format:
                    if not model_info or not model_info.supports_structured_output:
                        raise ValueError(f"Structured output is not supported for this model: `{self.model}`")  # noqa: E501
                    start_time = perf_counter()
                    completion = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        store=False,
                        **model_parameters,
                    )
                    end_time = perf_counter()
                    input_tokens = completion.usage.prompt_tokens
                    # these fields may not be available when using openai library for third-party
                    # providers
                    if (
                        hasattr(completion, 'usage')
                        and hasattr(completion.usage, 'prompt_tokens_details')
                        and hasattr(completion.usage.prompt_tokens_details, 'cached_tokens')
                    ):
                        cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
                        cached_cost = cached_tokens * pricing_lookup.get('cached', 0)
                        input_tokens -= cached_tokens  # remove cached tokens from input tokens
                    else:
                        cached_tokens = None
                        cached_cost = None

                    # Add timing to span
                    if span:
                        span.set_attribute("llm.request.duration", end_time - start_time)
                        span.set_attribute("llm.response.finish_reason", "completed")

                    yield StructuredOutputResponse(
                        parsed=completion.choices[0].message.parsed,
                        refusal=completion.choices[0].message.refusal,
                        input_tokens=input_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        cache_read_tokens=cached_tokens,
                        input_cost=completion.usage.prompt_tokens * pricing_lookup['input'],
                        output_cost=completion.usage.completion_tokens * pricing_lookup['output'],
                        cache_read_cost=cached_cost,
                        duration_seconds=end_time - start_time,
                    )
                else:
                    chunks = []
                    input_tokens = 0
                    output_tokens = 0
                    cached_tokens = 0
                    # `stream_options={'include_usage': True}` is required to get usage information
                    # from API when streaming.
                    # However, stream_options may not be valid when using openai library for
                    # third-party providers (i.e. if server_url is None we are using the openai
                    # library)
                    if (
                        self.server_url is None  # if using openai library
                        and (
                            # if either the user has not specified stream_options, or if they have
                            # and 'include_usage' is not set
                            'stream_options' not in model_parameters
                            or 'include_usage' not in model_parameters['stream_options']
                        )
                    ):
                        model_parameters['stream_options'] = {'include_usage': True}
                    start_time = perf_counter()
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        store=False,
                        **model_parameters,
                    )
                    async for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            parsed_chunk = _parse_completion_chunk(chunk)
                            yield parsed_chunk
                            chunks.append(parsed_chunk)
                        if hasattr(chunk, 'usage') and chunk.usage:
                            input_tokens += chunk.usage.prompt_tokens
                            output_tokens += chunk.usage.completion_tokens
                            # these fields may not be available when using openai library for
                            # third-party providers
                            if (
                                hasattr(chunk, 'usage')
                                and hasattr(chunk.usage, 'prompt_tokens_details')
                                and hasattr(chunk.usage.prompt_tokens_details, 'cached_tokens')
                            ):
                                cached_tokens += chunk.usage.prompt_tokens_details.cached_tokens
                    end_time = perf_counter()
                    input_tokens -= cached_tokens  # remove cached tokens from input tokens
                    if pricing_lookup:
                        input_cost = input_tokens * pricing_lookup['input']
                        output_cost = output_tokens * pricing_lookup['output']
                        cache_cost = cached_tokens * pricing_lookup.get('cached', 0)
                    else:
                        input_cost = None
                        output_cost = None
                        cache_cost = None

                    # Add timing to span
                    if span:
                        span.set_attribute("llm.request.duration", end_time - start_time)
                        span.set_attribute("llm.response.finish_reason", "completed")

                    yield TextResponse(
                        response=''.join([chunk.content for chunk in chunks]),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cached_tokens,
                        input_cost=input_cost,
                        output_cost=output_cost,
                        cache_read_cost=cache_cost,
                        duration_seconds=end_time - start_time,
                    )

            except Exception as e:
                # Add error information to span
                if span:
                    span.set_attribute("llm.request.error", str(e))
                    from opentelemetry import trace
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=str(e)),
                    )
                raise


def _tool_to_openai_schema(tool: Tool) -> dict[str, object]:
    """
    Convert the tool to the format expected by OpenAI API.

    OpenAI's function calling API has specific requirements for JSON Schema:
    - No default values are allowed anywhere in the schema
    - Nested objects require all properties to be listed as required
    - additionalProperties must be set to false
    """
    properties = {}
    required = []

    for param in tool.parameters:
        # Handle union types (any_of) by creating multiple schema options
        if param.any_of:
            any_of_schemas = []
            for union_type in param.any_of:
                json_type, extra_props = get_json_schema_type(union_type)
                # OpenAI API rejects schemas with default values, so we must remove them
                if 'default' in extra_props:
                    del extra_props['default']
                any_of_schemas.append({"type": json_type, **extra_props})
            param_dict = {"anyOf": any_of_schemas}
        else:
            # Convert Python types to JSON Schema types
            json_type, extra_props = get_json_schema_type(param.param_type)
            # OpenAI API rejects schemas with default values, so we must remove them
            if 'default' in extra_props:
                del extra_props['default']
            param_dict = {"type": json_type, **extra_props}

        # Add description to improve usability for the LLM
        if param.description:
            param_dict["description"] = param.description

        # Add enum/valid_values to constrain possible inputs
        if param.valid_values:
            param_dict["enum"] = param.valid_values

        # Special handling for object types:
        # 1. OpenAI requires additionalProperties: false
        # 2. For nested objects, all properties must be listed as required
        if json_type == 'object' and 'properties' in param_dict:
            # Prevent arbitrary properties from being added to objects
            param_dict['additionalProperties'] = False
            # OpenAI requires all properties of nested objects to be listed as required
            if param_dict['properties']:
                param_dict['required'] = list(param_dict['properties'].keys())

        properties[param.name] = param_dict

        # Only add genuinely required parameters to the top-level required list
        # This preserves the semantic meaning of "required" while maintaining compatibility
        if param.required:
            required.append(param.name)

    # Create the parameters object schema
    parameters_dict = {
        'type': 'object',
        'properties': properties,
    }
    if required:
        parameters_dict['required'] = required
    # Prevent arbitrary parameters from being added
    parameters_dict['additionalProperties'] = False

    # "strict" mode enforces all required parameters must be provided
    # Only enable it if all properties are required to avoid unnecessary constraints
    strict = all(param.required for param in tool.parameters or [])

    # Assemble the complete function schema
    result = {
        'type': 'function',
        'function': {
            'name': tool.name,
            'strict': strict,
            **({'description': tool.description} if tool.description else {}),
            'parameters': parameters_dict,
        },
    }
    # Remove any remaining default values that might be nested deeply in the schema
    # This ensures complete compatibility with OpenAI's requirements
    _remove_defaults_recursively(result)
    return result



@Client.register(RegisteredClients.OPENAI_TOOLS)
class OpenAITools(Client):
    """Wrapper for OpenAI API function/tool calling."""

    def __init__(
            self,
            model_name: str,
            tools: list[Tool],
            tool_choice: ToolChoice = ToolChoice.REQUIRED,
            server_url: str | None = None,
            api_key: str | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        Args:
            model_name:
                The model name to use for the API call (e.g. 'gpt-4').
            tools:
                List of Tool objects defining available tools.
            tool_choice:
                Controls if tools are required or optional.
            server_url:
                The base URL for the API call.
            api_key:
                The API key to use for the API call. If not provided, the API key will be read from
                the OPENAI_API_KEY environment variable.
            **model_kwargs:
                Additional parameters to pass to the API call
        """
        super().__init__(model_name, **model_kwargs)

        if (
            server_url is None  # using OpenAI provider
            and (
                model_name not in SUPPORTED_OPENAI_MODELS
                or not SUPPORTED_OPENAI_MODELS[model_name].supports_tools
            )
        ):
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.client = AsyncOpenAI(
            base_url=server_url,
            api_key=api_key or os.getenv('OPENAI_API_KEY') or 'None',
        )
        self.model = model_name
        self.model_parameters = model_kwargs or {}
        if 'temperature' not in self.model_parameters:
            self.model_parameters['temperature'] = 0.2

        if tool_choice == ToolChoice.REQUIRED:
            self.model_parameters['tool_choice'] = 'required'
        elif tool_choice == ToolChoice.AUTO:
            self.model_parameters['tool_choice'] = 'auto'
        else:
            raise ValueError(f"Invalid tool_choice: `{tool_choice}`")
        self.model_parameters['tools'] = [_tool_to_openai_schema(t) for t in tools]

    async def stream(  # noqa: PLR0912
            self,
            messages: list[dict[str, str]],
        ) -> AsyncGenerator[ToolPredictionResponse | None]:
        """
        Call the model with tools.

        Args:
            messages: List of messages to send to the model.
        """
        with safe_span(
            self.tracer,
            "llm.openai.tools",
            attributes={
                "llm.model": self.model,
                "llm.provider": "openai",
                "llm.operation": "tool_calling",
            },
        ) as span:
            model_info = SUPPORTED_OPENAI_MODELS.get(self.model)
            pricing_lookup = model_info.pricing if model_info else None

            # Add OpenAI-specific attributes
            if span:
                span.set_attribute("llm.api.vendor", "openai")
                if 'temperature' in self.model_parameters:
                    span.set_attribute("llm.request.temperature", self.model_parameters['temperature'])  # noqa: E501
                if 'tools' in self.model_parameters:
                    span.set_attribute("llm.tools.count", len(self.model_parameters['tools']))

            try:
                model_parameters = deepcopy(self.model_parameters)
                start = perf_counter()
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    store=False,
                    # i'm not sure it makes sense to stream chunks for tools, perhaps this will
                    # change in the future; but seems overly complicated for a tool call.
                    stream=False,
                    **model_parameters,
                )
                end = perf_counter()

                # Add timing to span
                if span:
                    span.set_attribute("llm.request.duration", end - start)
                    span.set_attribute("llm.response.finish_reason", "completed")

                # Calculate costs
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                # these fields may not be available when using openai library for third-party
                # providers
                cached_tokens = None
                cached_cost = None
                if (
                    hasattr(completion, 'usage')
                    and hasattr(completion.usage, 'prompt_tokens_details')
                    and hasattr(completion.usage.prompt_tokens_details, 'cached_tokens')
                ):
                    cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
                    if pricing_lookup:
                        cached_cost = cached_tokens * pricing_lookup.get('cached', 0)
                    input_tokens -= cached_tokens  # remove cached tokens from input tokens

                if pricing_lookup:
                    input_cost = input_tokens * pricing_lookup['input']
                    output_cost = output_tokens * pricing_lookup['output']

                if completion.choices[0].message.tool_calls:
                    message = None
                    tool_call = completion.choices[0].message.tool_calls[0]
                    tool_prediction = ToolPrediction(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                        call_id=tool_call.id,
                    )

                    # Add tool call info to span
                    if span:
                        span.set_attribute("llm.tool.name", tool_call.function.name)
                        span.set_attribute("llm.tool.arguments_count", len(tool_call.function.arguments))  # noqa: E501
                else:
                    message = completion.choices[0].message.content
                    tool_prediction = None

                    if span:
                        span.set_attribute("llm.response.type", "message")

                yield ToolPredictionResponse(
                    tool_prediction=tool_prediction,
                    message=message,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cached_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    cache_read_cost=cached_cost,
                    duration_seconds=end - start,
                )

            except Exception as e:
                # Add error information to span
                if span:
                    span.set_attribute("llm.request.error", str(e))
                    from opentelemetry import trace
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=str(e)),
                    )
                raise
