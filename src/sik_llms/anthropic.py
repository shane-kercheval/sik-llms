"""Helper functions for interacting with the Anthropic API."""
import os
import time
from collections.abc import AsyncGenerator
from anthropic import AsyncAnthropic, Anthropic as SyncAnthropic
from sik_llms.models_base import (
    Client,
    ChatChunkResponse,
    ChatResponseSummary,
    ContentType,
    Function,
    FunctionCallResponse,
    FunctionCallResult,
    RegisteredClients,
    ReasoningEffort,
    ToolChoice,
)


CHAT_MODEL_COST_PER_TOKEN = {
    'claude-3-7-sonnet-20250219': {'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000},

    'claude-3-5-haiku-20241022': {'input': 0.80 / 1_000_000, 'output': 4.0 / 1_000_000},
    'claude-3-5-sonnet-20241022': {'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000},

    'claude-3-opus-20240229': {'input': 15.00 / 1_000_000, 'output': 75.00 / 1_000_000},
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
        messages=[{"role": "user", "content": content}],
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


def _parse_completion_chunk(chunk) -> ChatChunkResponse | None:  # noqa: ANN001
    """Parse a chunk from the Anthropic API streaming response."""
    # Process content block deltas
    if chunk.type == 'content_block_start':  # noqa: SIM102
        if chunk.content_block.type == 'redacted_thinking':
            return ChatChunkResponse(
                content=chunk.content_block.data,
                content_type=ContentType.REDACTED_THINKING,
            )
    if chunk.type == 'content_block_delta':
        # Text delta
        if chunk.delta.type == 'text_delta':
            return ChatChunkResponse(
                content=chunk.delta.text,
                content_type=ContentType.TEXT,
            )
        # Thinking delta
        if chunk.delta.type == 'thinking_delta':
            return ChatChunkResponse(
                content=chunk.delta.thinking,
                content_type=ContentType.THINKING,
            )
    if chunk.type == 'error':
        return ChatChunkResponse(
            content=f"Error: type={chunk.error.type}, message={chunk.error.message}",
            content_type=ContentType.ERROR,
        )
    # All other event types are ignored (content_block_start, message_start, message_delta, etc.)
    return None


@Client.register(RegisteredClients.ANTHROPIC)
class Anthropic(Client):
    """
    Wrapper for Anthropic API which provides a simple interface for calling the
    messages.create method and parsing the response.
    """

    def __init__(
            self,
            model_name: str,
            max_tokens: int = 1_000,
            reasoning_effort: ReasoningEffort | None = None,
            thinking_budget_tokens: int | None = None,
            **model_kwargs: dict,
    ) -> None:
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


    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            else:
                anthropic_messages.append({'role': msg['role'], 'content': msg['content']})
        return system_content, anthropic_messages

    async def run_async(
            self,
            messages: list[dict],
        ) -> AsyncGenerator[ChatChunkResponse | ChatResponseSummary, None]:
        """
        Streams chat chunks and returns a final summary. Parameters passed here
        override those passed to the constructor.
        """
        system_content, anthropic_messages = self._convert_messages(messages)
        api_params = {
            'model': self.model,
            'messages': anthropic_messages,
            'stream': True,
            **self.model_parameters,
        }
        if system_content:
            api_params['system'] = system_content
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0

        chunks = []
        response = await self.client.messages.create(**api_params)
        async for chunk in response:
            if chunk.type == 'message_start':
                input_tokens = chunk.message.usage.input_tokens
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
            if chunk.content_type in (ContentType.TEXT, ContentType.THINKING):
                processed_chunks.append(chunk.content)

        yield ChatResponseSummary(
            content=''.join(processed_chunks),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['input'],
            output_cost=output_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['output'],
            duration_seconds=end_time - start_time,
        )


@Client.register(RegisteredClients.ANTHROPIC_FUNCTIONS)
class AnthropicFunctions(Client):
    """Wrapper for Anthropic API which provides a simple interface for using functions."""

    def __init__(
            self,
            model_name: str,
            functions: list[Function],
            tool_choice: ToolChoice = ToolChoice.REQUIRED,
            max_tokens: int = 1_000,
            **model_kwargs: dict,
    ) -> None:
        """
        Initialize the AnthropicFunctions client.

        Args:
            model_name:
                The model name to use for the API call (e.g. 'gpt-4').
            functions:
                List of Function objects defining available functions.
            tool_choice:
                Controls if tools are required or optional.
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

        tools = [func.to_anthropic() for func in functions]
        if tool_choice == ToolChoice.REQUIRED:
            tool_choice = 'any'
        elif tool_choice == ToolChoice.AUTO:
            tool_choice = 'auto'
        else:
            raise ValueError(f"Invalid tool_choice: `{tool_choice}`")
        self.model_parameters['tools'] = tools
        self.model_parameters['tool_choice'] = {'type': tool_choice}

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            else:
                anthropic_messages.append({'role': msg['role'], 'content': msg['content']})
        return system_content, anthropic_messages

    async def run_async(self, messages: list[dict]) -> FunctionCallResponse:
        """Runs the function call and returns the response."""
        system_content, anthropic_messages = self._convert_messages(messages)
        api_params = {
            'model': self.model,
            'messages': anthropic_messages,
            'stream': False,
            **self.model_parameters,
        }
        if system_content:
            api_params['system'] = system_content
        start_time = time.time()
        response = await self.client.messages.create(**api_params)
        end_time = time.time()

        function_call = None
        message = None
        if len(response.content) > 1:
            raise ValueError(f"Unexpected multiple content items in response: {response.content}")
        if response.content[0].type == 'tool_use':
            function_call = FunctionCallResult(
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
        return FunctionCallResponse(
            function_call=function_call,
            message=message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['input'],
            output_cost=output_tokens * CHAT_MODEL_COST_PER_TOKEN[self.model]['output'],
            duration_seconds=end_time - start_time,
        )
