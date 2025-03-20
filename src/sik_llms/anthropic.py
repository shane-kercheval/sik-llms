"""Helper functions for interacting with the Anthropic API."""
from copy import deepcopy
import os
import time
from collections.abc import AsyncGenerator
from anthropic import AsyncAnthropic, Anthropic as SyncAnthropic
from pydantic import BaseModel
from sik_llms.models_base import (
    Client,
    ErrorEvent,
    TextChunkEvent,
    TextResponse,
    ThinkingChunkEvent,
    Tool,
    ToolPredictionResponse,
    ToolPrediction,
    RegisteredClients,
    ReasoningEffort,
    StructuredOutputResponse,
    ToolChoice,
    pydantic_model_to_tool,
)


CHAT_MODEL_COST_PER_TOKEN = {
    'claude-3-7-sonnet-20250219': {
        'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000,
        'cache_write': 3.75 / 1_000_000, 'cache_read': 0.30 / 1_000_000,
    },
    'claude-3-5-sonnet-20241022': {
        'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000,
        'cache_write': 3.75 / 1_000_000, 'cache_read': 0.30 / 1_000_000,
    },
    'claude-3-5-haiku-20241022': {
        'input': 0.80 / 1_000_000, 'output': 4.0 / 1_000_000,
        'cache_write': 1.00 / 1_000_000, 'cache_read': 0.08 / 1_000_000,
    },

    'claude-3-opus-20240229': {
        'input': 15.00 / 1_000_000, 'output': 75.00 / 1_000_000,
        'cache_write': 18.75 / 1_000_000, 'cache_read': 1.50 / 1_000_000,
    },
}
CHAT_MODEL_COST_PER_TOKEN_LATEST = {
    'claude-3-7-sonnet-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-7-sonnet-20250219'],

    'claude-3-5-haiku-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-5-haiku-20241022'],
    'claude-3-5-sonnet-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-5-sonnet-20241022'],

    'claude-3-opus-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-opus-20240229'],
}
CHAT_MODEL_COST_PER_TOKEN.update(CHAT_MODEL_COST_PER_TOKEN_LATEST)


# Default thinking budget tokens for each reasoning effort level
REASONING_EFFORT_BUDGET = {
    ReasoningEffort.LOW: 4_000,
    ReasoningEffort.MEDIUM: 16_000,
    ReasoningEffort.HIGH: 32_000,
}


def num_tokens(model_name: str, content: str) -> int:
    """Returns the number of tokens for a given string."""
    client = SyncAnthropic()
    response = client.messages.count_tokens(
        model=model_name,
        messages=[{'role': 'user', 'content': content}],
    )
    return response.tokens


def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """Returns the number of tokens for a list of messages."""
    client = SyncAnthropic()
    response = client.messages.count_tokens(
        model=model_name,
        messages=messages,
    )
    return response.tokens


def _parse_completion_chunk(chunk) -> TextChunkEvent | None:  # noqa: ANN001
    """Parse a chunk from the Anthropic API streaming response."""
    # Process content block deltas
    if chunk.type == 'content_block_start':  # noqa: SIM102
        if chunk.content_block.type == 'redacted_thinking':
            return ThinkingChunkEvent(
                content=chunk.content_block.data,
                is_redacted=True,
            )
    if chunk.type == 'content_block_delta':
        # Text delta
        if chunk.delta.type == 'text_delta':
            return TextChunkEvent(
                content=chunk.delta.text,
            )
        # Thinking delta
        if chunk.delta.type == 'thinking_delta':
            return ThinkingChunkEvent(
                content=chunk.delta.thinking,
                is_redacted=False,
            )
    if chunk.type == 'error':
        return ErrorEvent(
            content=f"Error: type={chunk.error.type}, message={chunk.error.message}",
            metadata={
                'type': chunk.error.type,
                'message': chunk.error.message,
            },
        )
    # All other event types are ignored (content_block_start, message_start, message_delta, etc.)
    return None

def _convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert OpenAI-style messages to Anthropic format."""
    system_messages = []
    anthropic_messages = []

    for msg in deepcopy(messages):
        if msg['role'] == 'system':
            _ = msg.pop('role')
            text = msg.pop('content')
            system_messages.append({
                'type': 'text',
                'text': text,
                **msg,
            })
        else:
            anthropic_messages.append({'role': msg['role'], 'content': msg['content']})
    return system_messages, anthropic_messages


@Client.register(RegisteredClients.ANTHROPIC)
class Anthropic(Client):
    """
    Wrapper for Anthropic API which provides a simple interface for calling the
    messages.create method and parsing the response.
    """

    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            max_tokens: int = 1_000,
            reasoning_effort: ReasoningEffort | None = None,
            thinking_budget_tokens: int | None = None,
            response_format: type[BaseModel] | None = None,
            **model_kwargs: dict,
    ) -> None:
        if model_name not in CHAT_MODEL_COST_PER_TOKEN:
            raise ValueError(f"Model '{model_name}' is not supported.")
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model_name
        if not max_tokens:
            max_tokens = 1_000

        self.response_format = response_format
        self.model_parameters = {'max_tokens': max_tokens, **model_kwargs}
        # remove any None values
        self.model_parameters = {k: v for k, v in self.model_parameters.items() if v is not None}

        # Configure thinking based on reasoning_effort or thinking_budget_tokens
        thinking_config = None
        if reasoning_effort:
            thinking_budget = REASONING_EFFORT_BUDGET[reasoning_effort]
            self.model_parameters['max_tokens'] += thinking_budget
            thinking_config = {
                'type': 'enabled',
                'budget_tokens': thinking_budget,
            }
        elif thinking_budget_tokens:
            if thinking_budget_tokens < 1024:
                raise ValueError("thinking_budget_tokens must be at least 1024")
            self.model_parameters['max_tokens'] += thinking_budget_tokens
            thinking_config = {
                'type': 'enabled',
                'budget_tokens': thinking_budget_tokens,
            }

        if thinking_config:
            self.model_parameters['thinking'] = thinking_config
            # From docs: "Thinking isn't compatible with temperature, top_p, or top_k modifications
            # as well as forced tool use."
            if 'temperature' in self.model_parameters:
                self.model_parameters.pop('temperature')
            if 'top_p' in self.model_parameters:
                self.model_parameters.pop('top_p')
            if 'top_k' in self.model_parameters:
                self.model_parameters.pop('top_k')

    async def stream(
            self,
            messages: list[dict],
        ) -> AsyncGenerator[TextChunkEvent | TextResponse, None]:
        """
        Streams chat chunks and returns a final summary. Parameters passed here
        override those passed to the constructor.
        """
        if self.response_format:
            # Convert Pydantic model to Function
            function = pydantic_model_to_tool(self.response_format)
            # Create AnthropicFunctions client
            functions_client = AnthropicTools(
                model_name=self.model,
                tools=[function],
                tool_choice=ToolChoice.REQUIRED,
                max_tokens=self.model_parameters.get('max_tokens', 5000),
                temperature=0.2,
            )

            # Call functions client
            try:
                parsed = None
                refusal = None
                response = await functions_client.run_async(messages)
                # Extract function call result and convert to Pydantic model
                if response.tool_prediction:
                    try:
                        # Create instance of Pydantic model from arguments
                        parsed = self.response_format(**response.tool_prediction.arguments)
                    except Exception as e:
                        # If conversion fails, set refusal with error message
                        refusal=f"Failed to parse response: response={response}, error={e!s}"
                else:
                    # No function call, set refusal with model message
                    refusal=response.message
                # Yield the response
                yield StructuredOutputResponse(
                    parsed=parsed,
                    refusal=refusal,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    input_cost=response.input_cost,
                    output_cost=response.output_cost,
                    duration_seconds=response.duration_seconds,
                )
                return
            except Exception as e:
                # Handle any other errors
                yield ErrorEvent(
                    content=f"Error: {e!s}",
                    metadata={'error': e},
                )
                yield StructuredOutputResponse(
                    parsed=None,
                    refusal=f"Function call error: {e!s}",
                    input_tokens=0,
                    output_tokens=0,
                    input_cost=0,
                    output_cost=0,
                    duration_seconds=0,
                )
                return

        system_messages, anthropic_messages = _convert_messages(messages)
        api_params = {
            'model': self.model,
            'messages': anthropic_messages,
            'stream': True,
            **self.model_parameters,
        }
        if system_messages:
            api_params['system'] = system_messages
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

        chunks = []
        response = await self.client.messages.create(**api_params)
        async for chunk in response:
            if chunk.type == 'message_start':
                input_tokens += chunk.message.usage.input_tokens
                output_tokens += chunk.message.usage.output_tokens
                cache_creation_input_tokens += chunk.message.usage.cache_creation_input_tokens
                cache_read_input_tokens += chunk.message.usage.cache_read_input_tokens
            elif chunk.type == 'message_delta' and hasattr(chunk, 'usage'):
                output_tokens = chunk.usage.output_tokens
            parsed_chunk = _parse_completion_chunk(chunk)
            if parsed_chunk and parsed_chunk.content:
                yield parsed_chunk
                chunks.append(parsed_chunk)
        end_time = time.time()

        # Process content for summary based on content types
        processed_chunks = []
        for chunk in chunks:
            # Ignore REDACTED_THINKING chunks for the summary
            # if chunk.content_type in (ContentType.TEXT, ContentType.THINKING):
            if isinstance(chunk, TextChunkEvent) or (isinstance(chunk, ThinkingChunkEvent) and not chunk.is_redacted):  # noqa: E501
                processed_chunks.append(chunk.content)

        yield TextResponse(
            response=''.join(processed_chunks),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['input'],
            output_cost=output_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['output'],
            cache_write_tokens=cache_creation_input_tokens,
            cache_read_tokens=cache_read_input_tokens,
            cache_write_cost=cache_creation_input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['cache_write'],  # noqa: E501
            cache_read_cost=cache_read_input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['cache_read'],  # noqa: E501
            duration_seconds=end_time - start_time,
        )


@Client.register(RegisteredClients.ANTHROPIC_TOOLS)
class AnthropicTools(Client):
    """Wrapper for Anthropic API which provides a simple interface for using functions."""

    def __init__(
            self,
            model_name: str,
            tools: list[Tool],
            tool_choice: ToolChoice = ToolChoice.REQUIRED,
            cache_tools: bool = False,
            max_tokens: int = 1_000,
            **model_kwargs: dict,
    ) -> None:
        """
        Initialize the AnthropicFunctions client.

        Args:
            model_name:
                The model name to use for the API call (e.g. 'gpt-4').
            tools:
                List of Tool objects defining available tool.
            tool_choice:
                Controls if tools are required or optional.
            cache_tools:
                If True, caching will be used according to:
                https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#caching-tool-definitions
            max_tokens:
                The maximum number of tokens to generate in a single call.
            **model_kwargs:
                Additional parameters to pass to the API call
        """
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model_name
        if not max_tokens:
            max_tokens = 1_000

        self.model_parameters = {'max_tokens': max_tokens, **model_kwargs}
        # remove any None values
        self.model_parameters = {k: v for k, v in self.model_parameters.items() if v is not None}

        tools = [t.to_anthropic() for t in tools]
        if tool_choice == ToolChoice.REQUIRED:
            tool_choice = 'any'
        elif tool_choice == ToolChoice.AUTO:
            tool_choice = 'auto'
        else:
            raise ValueError(f"Invalid tool_choice: `{tool_choice}`")
        if cache_tools:
            # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#caching-tool-definitions
            tools[-1]['cache_control'] = {'type': 'ephemeral'}
        self.model_parameters['tools'] = tools
        self.model_parameters['tool_choice'] = {'type': tool_choice}

    async def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[ToolPredictionResponse | None]:  # noqa: E501
        """Runs the tool prediction and returns the response."""
        system_content, anthropic_messages = _convert_messages(messages)
        api_params = {
            'model': self.model,
            'messages': anthropic_messages,
            # i'm not sure it makes sense to stream chunks for tools, perhaps this will change
            # in the future; but seems overly complicated for a tool call.
            'stream': False,
            **self.model_parameters,
        }
        if system_content:
            api_params['system'] = system_content
        start_time = time.time()
        response = await self.client.messages.create(**api_params)
        end_time = time.time()

        tool_prediction = None
        message = None
        if len(response.content) > 1:
            raise ValueError(f"Unexpected multiple content items in response: {response.content}")
        if response.content[0].type == 'tool_use':
            tool_prediction = ToolPrediction(
                name=response.content[0].name,
                arguments=response.content[0].input,
                call_id=response.content[0].id,
            )
        elif response.content[0].type == 'text':
            message = response.content[0].text
        else:
            raise ValueError(f"Unexpected content type: {response.content[0].type}")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_creation_input_tokens = response.usage.cache_creation_input_tokens
        cache_read_input_tokens = response.usage.cache_read_input_tokens
        yield ToolPredictionResponse(
            tool_prediction=tool_prediction,
            message=message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['input'],
            output_cost=output_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['output'],
            cache_write_tokens=cache_creation_input_tokens,
            cache_read_tokens=cache_read_input_tokens,
            cache_write_cost=cache_creation_input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['cache_write'],  # noqa: E501
            cache_read_cost=cache_read_input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['cache_read'],  # noqa: E501
            duration_seconds=end_time - start_time,
        )
