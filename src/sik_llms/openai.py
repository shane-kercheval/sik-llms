"""Helper functions for OpenAI API."""
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
    Function,
    FunctionCallResponse,
    FunctionCallResult,
    Client,
    ChatChunkResponse,
    ChatResponseSummary,
    ReasoningEffort,
    RegisteredClients,
    StructuredOutputResponse,
    ToolChoice,
)

CHAT_MODEL_COST_PER_TOKEN = {
    # minor versions
    'gpt-4o-2024-05-13': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'gpt-4o-2024-08-06': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000},
    'gpt-4o-2024-11-20': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000},
    'gpt-4o-mini-2024-07-18':  {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
    'o1-2024-12-17': {'input': 15.00 / 1_000_000, 'output': 60.00 / 1_000_000},
    'o3-mini-2025-01-31': {'input': 1.10 / 1_000_000, 'output': 4.40 / 1_000_000},
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
        print("Warning: model not found. Using o200k_base encoding.")
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
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")  # noqa: E501
        return num_tokens_from_messages(messages=messages, model_name="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model_name:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")  # noqa: E501
        return num_tokens_from_messages(messages=messages, model_name="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model_name:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")  # noqa: E501
        return num_tokens_from_messages(messages=messages, model_name="gpt-4o-2024-08-06")
    elif "gpt-4" in model_name:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
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


def _parse_completion_chunk(chunk) -> ChatChunkResponse:  # noqa: ANN001
    assert chunk.object == 'chat.completion.chunk'
    log_prob = None
    if chunk.choices[0].logprobs:
        log_prob = chunk.choices[0].logprobs.content[0].logprob
    return ChatChunkResponse(
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
            model_name: str,
            server_url: str | None = None,
            reasoning_effort: ReasoningEffort | None = None,
            response_format: BaseModel | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        Args:
            client:
                An instance of the AsyncOpenAI client.
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            server_url:
                The base URL for the API call. Required for the `openai-compatible-server` model.
            reasoning_effort:
                The reasoning effort to use for the API call (reasoning model required).
            response_format:
                Pydantic class defining structure of the response (i.e. structured output)
            **model_kwargs: Additional parameters to pass to the API call
        """
        if model_name == 'openai-compatible-server':
            if not server_url:
                raise ValueError("Missing `server_url` for model `openai-compatible-server`")
            api_key = 'None'
            # If max_tokens is not provided, we'll set it;
            # otherwise the server will just output 1 token; not sure if this is llama.cpp issue
            # or something else
            if model_kwargs.get('max_tokens') is None:
                model_kwargs['max_tokens'] = -1
        else:
            server_url = None
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(f"Missing `api_key` for model `{model_name}`")

        self.client = AsyncOpenAI(base_url=server_url, api_key=api_key)
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
        # log_probs, temp, top_p are not supported with reasoning or reasoning models
        if reasoning_effort or any(self.model.startswith(prefix) for prefix in ['o1', 'o3']):
            self.log_probs = False
            self.model_parameters.pop('temperature', None)
            self.model_parameters.pop('top_p', None)
        else:
            self.log_probs = True

    async def run_async(
            self,
            messages: list[dict],
        ) -> AsyncGenerator[ChatChunkResponse | ChatResponseSummary | None]:
        """
        Streams chat chunks and returns a final summary. Note that any parameters passed to this
        method will override the parameters passed to the constructor.
        """
        if self.response_format:
            start_time = time.time()
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                store=False,
                **self.model_parameters,
            )
            end_time = time.time()
            yield ChatResponseSummary(
                content=StructuredOutputResponse(
                    parsed=completion.choices[0].message.parsed,
                    refusal=completion.choices[0].message.refusal,
                ),
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                input_cost=completion.usage.prompt_tokens * MODEL_COST_PER_TOKEN[self.model]['input'],  # noqa: E501
                output_cost=completion.usage.completion_tokens * MODEL_COST_PER_TOKEN[self.model]['output'],  # noqa: E501
                duration_seconds=end_time - start_time,
            )
        else:
            chunks = []
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                logprobs=self.log_probs,
                store=False,
                **self.model_parameters,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    parsed_chunk = _parse_completion_chunk(chunk)
                    yield parsed_chunk
                    chunks.append(parsed_chunk)
            end_time = time.time()
            if self.model == 'openai-compatible-server':
                input_tokens = len(str(messages)) // 4
                output_tokens = sum(len(chunk.content) for chunk in chunks) // 4
                total_input_cost=0
                total_output_cost=0
            else:
                input_tokens = num_tokens_from_messages(self.model, messages)
                output_tokens = sum(num_tokens(self.model, chunk.content) for chunk in chunks)
                total_input_cost=input_tokens * MODEL_COST_PER_TOKEN[self.model]['input']
                total_output_cost=output_tokens * MODEL_COST_PER_TOKEN[self.model]['output']
            yield ChatResponseSummary(
                content=''.join([chunk.content for chunk in chunks]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=total_input_cost,
                output_cost=total_output_cost,
                duration_seconds=end_time - start_time,
            )


@Client.register(RegisteredClients.OPENAI_FUNCTIONS)
class OpenAIFunctions(Client):
    """Wrapper for OpenAI API function calling."""

    def __init__(
            self,
            model_name: str,
            functions: list[Function],
            tool_choice: ToolChoice = ToolChoice.REQUIRED,
            server_url: str | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        Args:
            model_name:
                The model name to use for the API call (e.g. 'gpt-4').
            functions:
                List of Function objects defining available functions.
            tool_choice:
                Controls if tools are required or optional.
            server_url:
                The base URL for the API call. Required for the `openai-compatible-server` model.
            **model_kwargs:
                Additional parameters to pass to the API call
        """
        if model_name == 'openai-compatible-server':
            if not server_url:
                raise ValueError("Missing `server_url` for model `openai-compatible-server`")
            api_key = 'None'
        else:
            server_url = None
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(f"Missing `api_key` for model `{model_name}`")

        self.client = AsyncOpenAI(base_url=server_url, api_key=api_key)
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
        self.model_parameters['tools'] = [func.to_openai() for func in functions]

    async def run_async(self, messages: list[dict[str, str]]) -> FunctionCallResponse:
        """
        Call the model with functions.

        Args:
            messages: List of messages to send to the model.
            model_name: Optional model override.
            functions: Optional functions override.
            tool_choice: Controls if tools are required or optional.
            **model_kwargs: Additional parameters to override defaults.
        """
        start = time.time()
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            store=False,
            **self.model_parameters,
        )
        end = time.time()
        # Calculate costs
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        input_cost = input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['input']
        output_cost = output_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['output']

        if completion.choices[0].message.tool_calls:
            message = None
            tool_call = completion.choices[0].message.tool_calls[0]
            function_call = FunctionCallResult(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
                call_id=tool_call.id,
            )
        else:
            message = completion.choices[0].message.content
            function_call = None

        return FunctionCallResponse(
            function_call=function_call,
            message=message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            duration_seconds=end - start,
        )
