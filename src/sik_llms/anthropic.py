"""Helper functions for interacting with the Anthropic API."""
from copy import deepcopy
from datetime import date
import os
from time import perf_counter
from collections.abc import AsyncGenerator
from anthropic import AsyncAnthropic, Anthropic as SyncAnthropic
from pydantic import BaseModel
from sik_llms.models_base import (
    Client,
    ImageContent,
    ImageSourceType,
    ModelProvider,
    ModelInfo,
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
from sik_llms.utilities import get_json_schema_type
from sik_llms.telemetry import safe_span

ANTHROPIC_MODEL_LOOKUPS = [
    ModelInfo(
        model='claude-3-5-haiku-20241022',
        provider=ModelProvider.ANTHROPIC,
        max_output_tokens=8_192,
        context_window_size=200_000,
        pricing={
            'input': 0.80 / 1_000_000, 'output': 4.0 / 1_000_000,
            'cache_write': 1.00 / 1_000_000, 'cache_read': 0.08 / 1_000_000,
        },
        supports_tools=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2024, month=7, day=31),
    ),

    ModelInfo(
        model='claude-3-5-sonnet-20241022',
        provider=ModelProvider.ANTHROPIC,
        max_output_tokens=8_192,
        context_window_size=200_000,
        pricing={
            'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000,
            'cache_write': 3.75 / 1_000_000, 'cache_read': 0.30 / 1_000_000,
        },
        supports_tools=True,
        supports_images=True,
        knowledge_cutoff_date=date(year=2024, month=4, day=30),
    ),

    ModelInfo(
        model='claude-3-7-sonnet-20250219',
        provider=ModelProvider.ANTHROPIC,
        max_output_tokens=64_000,
        context_window_size=200_000,
        pricing={
            'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000,
            'cache_write': 3.75 / 1_000_000, 'cache_read': 0.30 / 1_000_000,
        },
        supports_tools=True,
        supports_images=True,
        supports_reasoning=True,
        knowledge_cutoff_date=date(year=2024, month=11, day=30),
        metadata={
            'max_output_extended_thinking': 64_000,
        },
    ),

    ModelInfo(
        model='claude-sonnet-4-20250514',
        provider=ModelProvider.ANTHROPIC,
        max_output_tokens=64_000,
        context_window_size=200_000,
        pricing={
            'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000,
            'cache_write': 3.75 / 1_000_000, 'cache_read': 0.30 / 1_000_000,
        },
        supports_tools=True,
        supports_images=True,
        supports_reasoning=True,
        knowledge_cutoff_date=date(year=2025, month=3, day=31),
        metadata={
            'max_output_extended_thinking': 64_000,
        },
    ),

    ModelInfo(
        model='claude-opus-4-20250514',
        provider=ModelProvider.ANTHROPIC,
        max_output_tokens=32_000,
        context_window_size=200_000,
        pricing={
            'input': 15.00 / 1_000_000, 'output': 75.00 / 1_000_000,
            'cache_write': 18.75 / 1_000_000, 'cache_read': 1.50 / 1_000_000,
        },
        supports_tools=True,
        supports_images=True,
        supports_reasoning=True,
        knowledge_cutoff_date=date(year=2025, month=3, day=31),
        metadata={
            'max_output_extended_thinking': 64_000,
        },
    ),
]
SUPPORTED_ANTHROPIC_MODELS = {model.model: model for model in ANTHROPIC_MODEL_LOOKUPS}
SUPPORTED_ANTHROPIC_MODELS['claude-3-5-haiku-latest'] = SUPPORTED_ANTHROPIC_MODELS['claude-3-5-haiku-20241022']  # noqa: E501
SUPPORTED_ANTHROPIC_MODELS['claude-3-5-sonnet-latest'] = SUPPORTED_ANTHROPIC_MODELS['claude-3-5-sonnet-20241022']  # noqa: E501
SUPPORTED_ANTHROPIC_MODELS['claude-3-7-sonnet-latest'] = SUPPORTED_ANTHROPIC_MODELS['claude-3-7-sonnet-20250219']  # noqa: E501


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


def _convert_messages(  # noqa: PLR0912
        messages: list[dict],
        cache_content: list[str] | str | None = None,
    ) -> tuple[list[dict], list[dict]]:
    """
    Convert OpenAI-style messages to Anthropic format.

    Args:
        messages:
            List of messages to convert.
        cache_content:
            List of strings to cache for the model. These strings are inserted into the system
            message and are marked with `"cache_control": {"type": "ephemeral"}` according to
            the documentation here:
            https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    """
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
        else:  # noqa: PLR5501
            if isinstance(msg['content'], list):
                content = []
                for item in msg['content']:
                    if isinstance(item, str):
                        content.append({
                            "type": "text",
                            "text": item.strip(),
                        })
                    elif isinstance(item, ImageContent):
                        if item.source_type == ImageSourceType.URL:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": item.data,
                                },
                            })
                        elif item.source_type == ImageSourceType.BASE64:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.media_type,
                                    "data": item.data,
                                },
                            })
                anthropic_messages.append({'role': msg['role'], 'content': content})
            else:
                anthropic_messages.append({'role': msg['role'], 'content': msg['content']})

    if cache_content:
        # Based on this example: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        # ```
        # system=[
        #     {
        #         "type": "text",
        #         "text": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n",  # noqa: E501
        #     },
        #     {
        #         "type": "text",
        #         "text": "<the entire contents of 'Pride and Prejudice'>",
        #         "cache_control": {"type": "ephemeral"}
        #     }
        # ],
        # ```
        if isinstance(cache_content, str):
            cache_content = [cache_content]

        for content in cache_content:
            system_messages.append({
                'type': 'text',
                'text': content.strip(),
                'cache_control': {'type': 'ephemeral'},
            })
    return system_messages, anthropic_messages


@Client.register(RegisteredClients.ANTHROPIC)
class Anthropic(Client):
    """
    Wrapper for Anthropic API which provides a simple interface for calling the
    messages.create method and parsing the response.
    """

    def __init__(  # noqa: PLR0912
            self,
            model_name: str,
            api_key: str | None = None,
            web_search: bool = False,
            max_web_searches: int = 5,
            reasoning_effort: ReasoningEffort | None = None,
            thinking_budget_tokens: int | None = None,
            response_format: type[BaseModel] | None = None,
            cache_content: list[str] | str | None = None,
            **model_kwargs: dict,
    ) -> None:
        """
        Initialize the Anthropic client.

        If not provided, `max_tokens` will be set to 8_000.

        Args:
            model_name:
                The model name to use for the API call (e.g. 'claude-3-7-sonnet-20250219').
            api_key:
                The API key to use for the API call. If not provided, the ANTHROPIC_API_KEY
                environment variable will be used.
            web_search:
                "The web search tool gives Claude direct access to real-time web content, allowing
                it to answer questions with up-to-date information beyond its knowledge cutoff.
                Claude automatically cites sources from search results as part of its answer."
                https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool
            max_web_searches:
                "The max_uses parameter limits the number of searches performed. If Claude attempts
                more searches than allowed, the web_search_tool_result will be an error with the
                max_uses_exceeded error code."
            reasoning_effort:
                Refers to the "thinking budget" refered to here:

                https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#implementing-extended-thinking

                ReasoningEffort is a enum with values LOW, MEDIUM, and HIGH that correspond to
                values are inspired by the documentation in the link above.

                You can provide the exact number of tokens to use for thinking by using the
                `thinking_budget_tokens` parameter instead.
            thinking_budget_tokens:
                The number of tokens to use for thinking. "The thinking budget is a target rather
                than a strict limit - actual token usage may vary based on the task."
            response_format:
                Allows the user to specify a Pydantic model which will be used to parse the
                response from the API. Behind the scenes, AnthropicTools is used to enable
                "Structured Output" similar to OpenAI's functionality.
            cache_content:
                List of strings to cache for the model. These strings are inserted into the system
                message and are marked with `"cache_control": {"type": "ephemeral"}` according to
                the documentation here:
                https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
            **model_kwargs:
                Additional parameters to pass to the API call
        """
        super().__init__(model_name, **model_kwargs)

        model_info = SUPPORTED_ANTHROPIC_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' is not supported.")
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model_name

        self.web_search = web_search
        self.max_web_searches = max_web_searches

        self.response_format = response_format
        # remove any None values
        self.model_parameters = {k: v for k, v in model_kwargs.items() if v is not None}
        if not self.model_parameters.get('max_tokens'):
            self.model_parameters['max_tokens'] = min(8_000, model_info.max_output_tokens)

        self.cache_content = cache_content

        # Configure thinking based on reasoning_effort or thinking_budget_tokens
        thinking_config = None
        if reasoning_effort or thinking_budget_tokens:
            if reasoning_effort and thinking_budget_tokens:
                raise ValueError("Only one of reasoning_effort or thinking_budget_tokens can be set.")  # noqa: E501
            if not model_info.supports_reasoning:
                raise ValueError(f"Model '{model_name}' does not support reasoning.")

            if reasoning_effort:
                # if reasoning_effort is set then thinking_budget_tokens is not set
                thinking_budget_tokens = REASONING_EFFORT_BUDGET[reasoning_effort]
            if thinking_budget_tokens < 1024:
                raise ValueError("thinking_budget_tokens must be at least 1024")

            thinking_tokens_limit = model_info.metadata.get('max_output_extended_thinking')
            if not thinking_tokens_limit:
                raise ValueError(f"Model '{model_name}' supports reasoning but does not have an extended thinking limit.")  # noqa: E501
            if thinking_budget_tokens > thinking_tokens_limit:
                raise ValueError(f"thinking_budget_tokens exceeds the model's extended thinking limit: {thinking_tokens_limit}")  # noqa: E501

            self.model_parameters['max_tokens'] += thinking_budget_tokens
            self.model_parameters['max_tokens'] = min(
                self.model_parameters['max_tokens'],
                thinking_tokens_limit,
            )
            # Add thinking budget to max_tokens
            thinking_config = {
                'type': 'enabled',
                'budget_tokens': thinking_budget_tokens,
            }
            self.model_parameters['thinking'] = thinking_config
            # From docs: "Thinking isn't compatible with temperature, top_p, or top_k modifications
            # as well as forced tool use."
            if 'temperature' in self.model_parameters:
                self.model_parameters.pop('temperature')
            if 'top_p' in self.model_parameters:
                self.model_parameters.pop('top_p')
            if 'top_k' in self.model_parameters:
                self.model_parameters.pop('top_k')


    async def stream(  # noqa: PLR0912, PLR0915
            self,
            messages: list[dict],
        ) -> AsyncGenerator[TextChunkEvent | TextResponse, None]:
        """
        Streams chat chunks and returns a final summary. Parameters passed here
        override those passed to the constructor.
        """
        with safe_span(
            self.tracer,
            "llm.anthropic.stream",
            attributes={
                "llm.model": self.model,
                "llm.provider": "anthropic",
                "llm.streaming": True,
            },
        ) as span:
            # Add Anthropic-specific attributes
            if span:
                span.set_attribute("llm.api.vendor", "anthropic")
                if hasattr(self, 'temperature') and 'temperature' in self.model_parameters:
                    span.set_attribute("llm.request.temperature", self.model_parameters['temperature'])  # noqa: E501
                if hasattr(self, 'max_tokens') and 'max_tokens' in self.model_parameters:
                    span.set_attribute("llm.request.max_tokens", self.model_parameters['max_tokens'])  # noqa: E501
                if self.web_search:
                    span.set_attribute("llm.web_search.enabled", True)
                    span.set_attribute("llm.web_search.max_uses", self.max_web_searches)

            start_time = perf_counter()

            try:
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
                                refusal=f"Failed to parse response: response={response}, error={e!s}"  # noqa: E501
                        else:
                            # No function call, set refusal with model message
                            refusal=response.message

                        # Add timing to span
                        if span:
                            span.set_attribute("llm.request.duration", perf_counter() - start_time)
                            span.set_attribute("llm.response.finish_reason", "completed")

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
                        if span:
                            span.set_attribute("llm.request.error", str(e))
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

                system_messages, anthropic_messages = _convert_messages(
                    messages,
                    cache_content=self.cache_content,
                )
                api_params = {
                    'model': self.model,
                    'messages': anthropic_messages,
                    'stream': True,
                    **self.model_parameters,
                }
                if system_messages:
                    api_params['system'] = system_messages

                if self.web_search:
                    api_params['tools'] = [{
                        'type': 'web_search_20250305',
                        'name': 'web_search',
                        'max_uses': self.max_web_searches,
                    }]

                input_tokens = 0
                output_tokens = 0
                cache_creation_input_tokens = 0
                cache_read_input_tokens = 0

                start_time = perf_counter()
                chunks = []
                response = await self.client.messages.create(**api_params)
                async for chunk in response:
                    if chunk.type == 'message_start':
                        input_tokens += chunk.message.usage.input_tokens
                        output_tokens += chunk.message.usage.output_tokens
                        cache_creation_input_tokens += chunk.message.usage.cache_creation_input_tokens  # noqa: E501
                        cache_read_input_tokens += chunk.message.usage.cache_read_input_tokens
                    elif chunk.type == 'message_delta' and hasattr(chunk, 'usage'):
                        output_tokens = chunk.usage.output_tokens
                    parsed_chunk = _parse_completion_chunk(chunk)
                    if parsed_chunk and parsed_chunk.content:
                        yield parsed_chunk
                        chunks.append(parsed_chunk)
                end_time = perf_counter()

                # Add timing to span
                if span:
                    span.set_attribute("llm.request.duration", end_time - start_time)
                    span.set_attribute("llm.response.finish_reason", "completed")

                # Process content for summary based on content types
                processed_chunks = []
                for chunk in chunks:
                    # Ignore REDACTED_THINKING chunks for the summary
                    # if chunk.content_type in (ContentType.TEXT, ContentType.THINKING):
                    if isinstance(chunk, TextChunkEvent) or (isinstance(chunk, ThinkingChunkEvent) and not chunk.is_redacted):  # noqa: E501
                        processed_chunks.append(chunk.content)

                pricing_lookup = SUPPORTED_ANTHROPIC_MODELS[self.model].pricing
                yield TextResponse(
                    response=''.join(processed_chunks),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_cost=input_tokens * pricing_lookup['input'],
                    output_cost=output_tokens * pricing_lookup['output'],
                    cache_write_tokens=cache_creation_input_tokens,
                    cache_read_tokens=cache_read_input_tokens,
                    cache_write_cost=cache_creation_input_tokens * pricing_lookup['cache_write'],
                    cache_read_cost=cache_read_input_tokens * pricing_lookup['cache_read'],
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


def _tool_to_anthropic_schema(tool: Tool) -> dict[str, object]:  # noqa: PLR0912
    """
    Convert the tool to the format expected by Anthropic API.

    Follows JSON Schema best practices:
    - Removes default values for consistency with OpenAI implementation
    - Sets additionalProperties: false to prevent unexpected properties
    - Preserves dictionary type constraints for proper typing
    """
    properties = {}
    required = []

    for param in tool.parameters:
        # Handle union types (any_of) by creating multiple schema options
        if param.any_of:
            any_of_schemas = []
            for union_type in param.any_of:
                json_type, extra_props = get_json_schema_type(union_type)
                # Remove any default values from extra_props
                if 'default' in extra_props:
                    del extra_props['default']
                any_of_schemas.append({"type": json_type, **extra_props})
            param_dict = {"anyOf": any_of_schemas}
        else:
            # Convert Python types to JSON Schema types
            json_type, extra_props = get_json_schema_type(param.param_type)
            # Remove any default values from extra_props
            if 'default' in extra_props:
                del extra_props['default']
            param_dict = {"type": json_type, **extra_props}

        # Add description to improve usability for the LLM
        if param.description:
            param_dict["description"] = param.description

        # Add enum values if provided
        if param.valid_values:
            param_dict["enum"] = param.valid_values

        # Special handling for object types:
        if json_type == 'object':
            # If this is a dictionary type with value type constraints, preserve them
            if 'additionalProperties' in extra_props and isinstance(extra_props['additionalProperties'], dict):  # noqa: E501
                param_dict['additionalProperties'] = extra_props['additionalProperties']
            else:
                param_dict['additionalProperties'] = False

            # For nested objects, ensure all properties are required (similar to OpenAI)
            if param_dict.get('properties'):
                param_dict['required'] = list(param_dict['properties'].keys())

        properties[param.name] = param_dict
        if param.required:
            required.append(param.name)

    parameters_dict = {
        'type': 'object',
        'properties': properties,
    }
    if required:
        parameters_dict['required'] = required
    parameters_dict['additionalProperties'] = False

    # Assemble the complete tool schema
    result = {
        'name': tool.name,
        **({'description': tool.description} if tool.description else {}),
        'input_schema': parameters_dict,
    }

    # # Remove any remaining default values that might be nested deeply in the schema
    # _remove_defaults_recursively(result)
    return result  # noqa: RET504


@Client.register(RegisteredClients.ANTHROPIC_TOOLS)
class AnthropicTools(Client):
    """Wrapper for Anthropic API which provides a simple interface for using functions."""

    def __init__(
            self,
            model_name: str,
            tools: list[Tool],
            tool_choice: ToolChoice = ToolChoice.REQUIRED,
            cache_tools: bool = False,
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
            **model_kwargs:
                Additional parameters to pass to the API call
        """
        super().__init__(model_name, **model_kwargs)

        model_info = SUPPORTED_ANTHROPIC_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' is not supported.")

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model_name

        self.model_parameters = {k: v for k, v in model_kwargs.items() if v is not None}
        if not self.model_parameters.get('max_tokens'):
            self.model_parameters['max_tokens'] = min(8_000, model_info.max_output_tokens)

        tools = [_tool_to_anthropic_schema(t) for t in tools]
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

    async def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[ToolPredictionResponse | None]:  # noqa: E501, PLR0912
        """Runs the tool prediction and returns the response."""
        with safe_span(
            self.tracer,
            "llm.anthropic.tools",
            attributes={
                "llm.model": self.model,
                "llm.provider": "anthropic",
                "llm.operation": "tool_calling",
            },
        ) as span:
            # Add Anthropic-specific attributes
            if span:
                span.set_attribute("llm.api.vendor", "anthropic")
                if 'temperature' in self.model_parameters:
                    span.set_attribute("llm.request.temperature", self.model_parameters['temperature'])  # noqa: E501
                if 'tools' in self.model_parameters:
                    span.set_attribute("llm.tools.count", len(self.model_parameters['tools']))

            try:
                system_content, anthropic_messages = _convert_messages(messages)
                api_params = {
                    'model': self.model,
                    'messages': anthropic_messages,
                    # i'm not sure it makes sense to stream chunks for tools, perhaps this will
                    # change in the future; but seems overly complicated for a tool call.
                    'stream': False,
                    **self.model_parameters,
                }
                if system_content:
                    api_params['system'] = system_content
                start_time = perf_counter()
                response = await self.client.messages.create(**api_params)
                end_time = perf_counter()

                # Add timing to span
                if span:
                    span.set_attribute("llm.request.duration", end_time - start_time)
                    span.set_attribute("llm.response.finish_reason", "completed")

                tool_prediction = None
                message = None
                if len(response.content) > 1:
                    raise ValueError(f"Unexpected multiple content items in response: {response.content}")  # noqa: E501
                if response.content[0].type == 'tool_use':
                    tool_prediction = ToolPrediction(
                        name=response.content[0].name,
                        arguments=response.content[0].input,
                        call_id=response.content[0].id,
                    )

                    # Add tool call info to span
                    if span:
                        span.set_attribute("llm.tool.name", response.content[0].name)
                        span.set_attribute("llm.tool.arguments_count", len(response.content[0].input))  # noqa: E501
                elif response.content[0].type == 'text':
                    message = response.content[0].text

                    if span:
                        span.set_attribute("llm.response.type", "message")
                else:
                    raise ValueError(f"Unexpected content type: {response.content[0].type}")

                pricing_lookup = SUPPORTED_ANTHROPIC_MODELS[self.model].pricing
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cache_creation_input_tokens = response.usage.cache_creation_input_tokens
                cache_read_input_tokens = response.usage.cache_read_input_tokens
                yield ToolPredictionResponse(
                    tool_prediction=tool_prediction,
                    message=message,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_cost=input_tokens * pricing_lookup['input'],
                    output_cost=output_tokens * pricing_lookup['output'],
                    cache_write_tokens=cache_creation_input_tokens,
                    cache_read_tokens=cache_read_input_tokens,
                    cache_write_cost=cache_creation_input_tokens * pricing_lookup['cache_write'],
                    cache_read_cost=cache_read_input_tokens * pricing_lookup['cache_read'],
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
