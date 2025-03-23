"""Helper functions for OpenAI API."""
from copy import deepcopy
from datetime import date
from functools import cache
import json
import os
import time
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
load_dotenv()

# Define all OpenAI models
OPENAI_MODEL_INFOS = [
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
SUPPORTED_OPENAI_MODELS['gpt-4o-mini'] = SUPPORTED_OPENAI_MODELS['gpt-4o-mini-2024-07-18']
SUPPORTED_OPENAI_MODELS['gpt-4o'] = SUPPORTED_OPENAI_MODELS['gpt-4o-2024-11-20']
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

    async def stream(
            self,
            messages: list[dict],
        ) -> AsyncGenerator[TextChunkEvent | TextResponse | None]:
        """
        Streams chat chunks and returns a final summary. Note that any parameters passed to this
        method will override the parameters passed to the constructor.
        """
        model_info = SUPPORTED_OPENAI_MODELS.get(self.model)
        pricing_lookup = model_info.pricing if model_info else None

        messages = _convert_messages(messages, self.cache_content)
        model_parameters = deepcopy(self.model_parameters)
        if self.response_format:
            if not model_info or not model_info.supports_structured_output:
                raise ValueError(f"Structured output is not supported for this model: `{self.model}`")  # noqa: E501
            start_time = time.time()
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                store=False,
                **model_parameters,
            )
            end_time = time.time()
            input_tokens = completion.usage.prompt_tokens
            # these fields may not be available when using openai library for third-party providers
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
            # However, stream_options may not be valid when using openai library for third-party
            # providers (i.e. if server_url is None we are using the openai library)
            if (
                self.server_url is None  # if using openai library
                and (
                    # if either the user has not specified stream_options, or if they have and
                    # 'include_usage' is not set
                    'stream_options' not in model_parameters
                    or 'include_usage' not in model_parameters['stream_options']
                )
            ):
                model_parameters['stream_options'] = {'include_usage': True}
            start_time = time.time()
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
                    # these fields may not be available when using openai library for third-party
                    # providers
                    if (
                        hasattr(chunk, 'usage')
                        and hasattr(chunk.usage, 'prompt_tokens_details')
                        and hasattr(chunk.usage.prompt_tokens_details, 'cached_tokens')
                    ):
                        cached_tokens += chunk.usage.prompt_tokens_details.cached_tokens
            end_time = time.time()
            input_tokens -= cached_tokens  # remove cached tokens from input tokens
            if pricing_lookup:
                input_cost = input_tokens * pricing_lookup['input']
                output_cost = output_tokens * pricing_lookup['output']
                cache_cost = cached_tokens * pricing_lookup.get('cached', 0)
            else:
                input_cost = None
                output_cost = None
                cache_cost = None
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
        self.model_parameters['tools'] = [t.to_openai() for t in tools]

    async def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[ToolPredictionResponse | None]:  # noqa: E501
        """
        Call the model with tools.

        Args:
            messages: List of messages to send to the model.
        """
        model_info = SUPPORTED_OPENAI_MODELS.get(self.model)
        pricing_lookup = model_info.pricing if model_info else None

        model_parameters = deepcopy(self.model_parameters)
        start = time.time()
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            store=False,
            # i'm not sure it makes sense to stream chunks for tools, perhaps this will change
            # in the future; but seems overly complicated for a tool call.
            stream=False,
            **model_parameters,
        )
        end = time.time()
        # Calculate costs
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        # these fields may not be available when using openai library for third-party providers
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
        else:
            message = completion.choices[0].message.content
            tool_prediction = None

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
