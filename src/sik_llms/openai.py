"""Helper functions for OpenAI API."""
from copy import deepcopy
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


CHAT_MODEL_COST_PER_TOKEN = {
    # minor versions
    'gpt-4o-2024-05-13': {
        'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000,
        'cached': 1.25 / 1_000_000,
    },
    'gpt-4o-2024-08-06': {
        'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000,
        'cached': 1.25 / 1_000_000,
    },
    'gpt-4o-2024-11-20': {
        'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000,
        'cached': 1.25 / 1_000_000,
    },
    'gpt-4o-mini-2024-07-18':  {
        'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000,
        'cached': 0.075 / 1_000_000,
    },
    'o1-2024-12-17': {
        'input': 15.00 / 1_000_000, 'output': 60.00 / 1_000_000,
        'cached': 7.50 / 1_000_000,
    },
    'o3-mini-2025-01-31': {
        'input': 1.10 / 1_000_000, 'output': 4.40 / 1_000_000,
        'cached': 0.55 / 1_000_000,
    },
    # LEGACY MODELS
    'gpt-4-turbo': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-4-turbo-2024-04-09': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-4-0125-preview': {'input': 0.01 / 1_000, 'output': 0.03 / 1_000},
    'gpt-3.5-turbo': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},
    'gpt-3.5-turbo-0125': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},
    'gpt-4-0613': {'input': 0.03 / 1_000, 'output': 0.06 / 1_000},
}
CHAT_MODEL_COST_PER_TOKEN_PRIMARY = {
    'gpt-4o-mini': CHAT_MODEL_COST_PER_TOKEN['gpt-4o-mini-2024-07-18'],
    'gpt-4o': CHAT_MODEL_COST_PER_TOKEN['gpt-4o-2024-11-20'],
    'o1': CHAT_MODEL_COST_PER_TOKEN['o1-2024-12-17'],
    'o3-mini': CHAT_MODEL_COST_PER_TOKEN['o3-mini-2025-01-31'],
}
CHAT_MODEL_COST_PER_TOKEN.update(CHAT_MODEL_COST_PER_TOKEN_PRIMARY)


EMBEDDING_MODEL_COST_PER_TOKEN = {
    # "Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens
    # is about 750 words. This paragraph is 35 tokens."
    # https://openai.com/pricing
    # https://platform.openai.com/docs/models
    ####
    # Embedding models
    ####
    # LATEST MODELS
    # https://openai.com/blog/new-embedding-models-and-api-updates
    'text-embedding-3-small': 0.02 / 1_000_000,
    'text-embedding-3-large': 0.13 / 1_000_000,
    # LEGACY MODELS
    'text-embedding-ada-002': 0.1 / 1_000_000,
}

MODEL_COST_PER_TOKEN = CHAT_MODEL_COST_PER_TOKEN | EMBEDDING_MODEL_COST_PER_TOKEN


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
    assert chunk.object == 'chat.completion.chunk'
    log_prob = None
    if chunk.choices[0].logprobs:
        log_prob = chunk.choices[0].logprobs.content[0].logprob
    return TextChunkEvent(
        content=chunk.choices[0].delta.content,
        logprob=log_prob,
    )


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
            server_url: str | None = None,
            api_key: str | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        Args:
            client:
                An instance of the AsyncOpenAI client.
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            reasoning_effort:
                The reasoning effort to use for the API call (reasoning model required).
            response_format:
                Pydantic class defining structure of the response (i.e. structured output)
            server_url:
                The base URL for the API call.
            api_key:
                The API key to use for the API call. If not provided, the API key will be read from
                the OPENAI_API_KEY environment variable.
            **model_kwargs: Additional parameters to pass to the API call
        """
        if server_url is None and model_name not in MODEL_COST_PER_TOKEN:
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

    async def stream(
            self,
            messages: list[dict],
        ) -> AsyncGenerator[TextChunkEvent | TextResponse | None]:
        """
        Streams chat chunks and returns a final summary. Note that any parameters passed to this
        method will override the parameters passed to the constructor.
        """
        model_parameters = deepcopy(self.model_parameters)
        if self.response_format:
            start_time = time.time()
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                store=False,
                **model_parameters,
            )
            end_time = time.time()
            # these fields may not be available when using openai library for third-party providers
            if (
                hasattr(completion, 'usage')
                and hasattr(completion.usage, 'prompt_tokens_details')
                and hasattr(completion.usage.prompt_tokens_details, 'cached_tokens')
            ):
                cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
                cached_cost = cached_tokens * MODEL_COST_PER_TOKEN[self.model].get('cached', 0)
            else:
                cached_tokens = None
                cached_cost = None
            yield StructuredOutputResponse(
                parsed=completion.choices[0].message.parsed,
                refusal=completion.choices[0].message.refusal,
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                cache_read_tokens=cached_tokens,
                input_cost=completion.usage.prompt_tokens * MODEL_COST_PER_TOKEN[self.model]['input'],  # noqa: E501
                output_cost=completion.usage.completion_tokens * MODEL_COST_PER_TOKEN[self.model]['output'],  # noqa: E501
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
            if self.model in MODEL_COST_PER_TOKEN:
                input_cost = input_tokens * MODEL_COST_PER_TOKEN[self.model]['input']
                output_cost = output_tokens * MODEL_COST_PER_TOKEN[self.model]['output']
                cache_cost = cached_tokens * MODEL_COST_PER_TOKEN[self.model].get('cached', 0)
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
        if server_url is None and model_name not in MODEL_COST_PER_TOKEN:
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
        input_cost = input_tokens * MODEL_COST_PER_TOKEN[self.model]['input']
        output_cost = output_tokens * MODEL_COST_PER_TOKEN[self.model]['output']

        # these fields may not be available when using openai library for third-party providers
        if (
            hasattr(completion, 'usage')
            and hasattr(completion.usage, 'prompt_tokens_details')
            and hasattr(completion.usage.prompt_tokens_details, 'cached_tokens')
        ):
            cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
            cached_cost = cached_tokens * MODEL_COST_PER_TOKEN[self.model].get('cached', 0)
        else:
            cached_tokens = None
            cached_cost = None

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
