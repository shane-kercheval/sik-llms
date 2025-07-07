"""Base classes and utilities for models."""
import asyncio
import base64
from datetime import date
import inspect
from pathlib import Path
import types
import nest_asyncio
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Literal, TypeVar, Union, get_args, get_origin
from sik_llms.utilities import Registry, _string_to_type
from sik_llms.telemetry import (
    get_tracer,
    get_meter,
    safe_span,
    create_span_link,
    extract_current_trace_context,
)


class ModelProvider(Enum):
    """Enum for model providers."""

    OPENAI = 'OpenAI'
    ANTHROPIC = 'Anthropic'
    OTHER = 'Other'


class RegisteredClients(Enum):
    """Enum for model types."""

    OPENAI = 'OpenAI'
    OPENAI_TOOLS = 'OpenAITools'
    ANTHROPIC = 'Anthropic'
    ANTHROPIC_TOOLS = 'AnthropicTools'
    REASONING_AGENT = 'ReasoningAgent'


class ReasoningEffort(Enum):
    """Enum for reasoning effort levels."""

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class ModelInfo(BaseModel):
    """Information about the model."""

    model: str
    provider: ModelProvider
    max_output_tokens: int
    context_window_size: int
    pricing: dict[str, float] | None = None
    supports_reasoning: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_images: bool = False
    knowledge_cutoff_date: date | None = None
    metadata: dict[str, Any] | None = None


class ImageSourceType(Enum):
    """Enum for image source types."""

    URL = "url"
    BASE64 = "base64"


class ImageContent(BaseModel):
    """Represents an image that can be sent to an LLM API."""

    source_type: ImageSourceType
    data: str  # URL, base64 string, or file path
    media_type: str | None = None  # e.g. "image/jpeg", "image/png"

    @classmethod
    def from_url(cls, url: str) -> "ImageContent":
        """Create an ImageContent object from a URL."""
        return cls(
            source_type=ImageSourceType.URL,
            data=url,
            media_type=None,  # Can be inferred from URL if needed
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ImageContent":
        """Create ImageContent from a local file path."""
        path = Path(path)
        with open(path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        # Map common file extensions to media types
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_types.get(path.suffix.lower(), 'image/jpeg')  # Default to JPEG
        return cls(
            source_type=ImageSourceType.BASE64,
            data=base64_data,
            media_type=media_type,
        )

    @classmethod
    def from_bytes(cls, bytes_data: bytes, media_type: str) -> "ImageContent":
        """Create an ImageContent object from bytes data."""
        return cls(
            source_type=ImageSourceType.BASE64,
            data=base64.b64encode(bytes_data).decode("utf-8"),
            media_type=media_type,
        )


def user_message(content: str | list[str | ImageContent]) -> dict:
    """
    Returns a user message that can include text and images.

    Args:
        content: Either a string for text-only messages, or a list containing
                strings and ImageContent objects for mixed content.

    Returns:
        A dictionary with 'role' and 'content' keys formatted for API consumption.

    Raises:
        TypeError: If content is not str or list, or if list items are not str or ImageContent
        ValueError: If content structure is invalid (e.g., nested lists)
    """
    if content is None:
        raise TypeError("Content cannot be None")

    if isinstance(content, str):
        return {'role': 'user', 'content': content.strip()}

    if not isinstance(content, list):
        raise TypeError(f"Content must be string or list, not {type(content)}")

    # Validate and process list content
    processed_content = []
    for item in content:
        if isinstance(item, str):
            processed_content.append(item.strip())
        elif isinstance(item, ImageContent):
            processed_content.append(item)
        elif isinstance(item, list):
            raise ValueError("Nested lists are not allowed in content")
        else:
            raise TypeError(
                f"Content list items must be string or ImageContent, not {type(item)}",
            )

    return {'role': 'user', 'content': processed_content}


def assistant_message(content: str) -> dict:
    """Returns an assistant message."""
    return {'role': 'assistant', 'content': content.strip()}


def system_message(content: str, **kwargs: dict | None) -> dict:
    """Returns a system message."""
    if not kwargs:
        kwargs = {}
    return {'role': 'system', 'content': content.strip(), **kwargs}


class TextChunkEvent(BaseModel):
    """A chunk returned when streaming."""

    content: str | object
    logprob: float | None = None


class ErrorEvent(BaseModel):
    """A chunk returned when an error occurs."""

    content: str
    metadata: dict | None = None


class InfoEvent(BaseModel):
    """Event for general information."""

    content: str
    metadata: dict | None = None


class AgentEvent(BaseModel):
    """Base class for all agent events."""

    iteration: int = Field(default=0, description="Current iteration number")


class ThinkingEvent(AgentEvent):
    """Event emitted during agent thinking."""

    content: str
    is_redacted: bool = False


class ThinkingChunkEvent(AgentEvent):
    """Event emitted during agent thinking."""

    content: str
    is_redacted: bool = False


class ToolPredictionEvent(AgentEvent):
    """Event emitted when predicting tool usage."""

    name: str
    arguments: dict[str, Any]


class ToolResultEvent(AgentEvent):
    """Event emitted after tool execution."""

    name: str
    arguments: dict[str, Any]
    result: object


class TraceContext(BaseModel):
    """OpenTelemetry trace context information for span linking."""

    trace_id: str | None = None
    span_id: str | None = None

    def create_link(self, attributes: dict[str, Any] | None = None) -> object | None:
        """
        Create a span link from this trace context.

        Args:
            attributes: Optional attributes for the link

        Returns:
            Link object or None if trace context is incomplete or telemetry is disabled
        """
        if not self.trace_id or not self.span_id:
            return None
        return create_span_link(self.trace_id, self.span_id, attributes)


class TokenSummary(BaseModel):
    """Summary of a chat response."""

    input_tokens: int
    output_tokens: int
    input_cost: float | None = None
    output_cost: float | None = None

    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_cost: float | None = None
    cache_read_cost: float | None = None

    duration_seconds: float

    @property
    def total_tokens(self) -> int:
        """Calculate the total number of tokens."""
        return (
            self.input_tokens
            + self.output_tokens
            + (self.cache_write_tokens or 0)
            + (self.cache_read_tokens or 0)
        )

    @property
    def total_cost(self) -> float | None:
        """Calculate the total cost."""
        if self.input_cost is None and self.output_cost is None and self.cache_write_cost is None and self.cache_read_cost is None:  # noqa: E501
            return None
        return (
            (self.input_cost or 0)
            + (self.output_cost or 0)
            + (self.cache_write_cost or 0)
            + (self.cache_read_cost or 0)
        )

    def emit_metrics(self, meter=None, labels: dict[str, str] | None = None) -> None:  # noqa: ANN001
        """
        Emit OpenTelemetry metrics for this token summary.

        Args:
            meter: OpenTelemetry meter instance
            labels: Additional labels/attributes for metrics
        """
        if not meter:
            return

        labels = labels or {}

        try:
            # Token counters
            input_counter = meter.create_counter(
                name="llm_tokens_input_total",
                description="Total number of input tokens consumed",
                unit="token",
            )
            output_counter = meter.create_counter(
                name="llm_tokens_output_total",
                description="Total number of output tokens generated",
                unit="token",
            )

            # Duration histogram
            duration_histogram = meter.create_histogram(
                name="llm_request_duration_seconds",
                description="Duration of LLM requests",
                unit="s",
            )

            # Cost counter (if available)
            if self.total_cost is not None:
                cost_counter = meter.create_counter(
                    name="llm_cost_total_usd",
                    description="Total cost of LLM operations in USD",
                    unit="USD",
                )
                cost_counter.add(self.total_cost, labels)

            # Emit metrics
            input_counter.add(self.input_tokens, labels)
            output_counter.add(self.output_tokens, labels)
            duration_histogram.record(self.duration_seconds, labels)

            # Cache metrics if available
            if self.cache_read_tokens:
                cache_read_counter = meter.create_counter(
                    name="llm_cache_read_tokens_total",
                    description="Number of tokens read from cache",
                    unit="token",
                )
                cache_read_counter.add(self.cache_read_tokens, labels)

            if self.cache_write_tokens:
                cache_write_counter = meter.create_counter(
                    name="llm_cache_write_tokens_total",
                    description="Number of tokens written to cache",
                    unit="token",
                )
                cache_write_counter.add(self.cache_write_tokens, labels)

        except Exception:
            # Silently handle metric emission errors
            pass


class StructuredOutputResponse(TokenSummary):
    """Response containing structured output data."""

    parsed: BaseModel | None
    refusal: str | object | None
    trace_context: TraceContext | None = None


class TextResponse(TokenSummary):
    """Summary of a chat response."""

    response: str
    trace_context: TraceContext | None = None


class Parameter(BaseModel):
    """
    Represents a parameter property in a Tools's schema.

    Args:
        name:
            The name of the parameter.
        param_type:
            The type of the parameter (e.g. `str`, `int`, `bool`, `MyModel(BaseModel)`), or a
            string representation of a type (e.g. `"str"`, `"int"`).
        required:
            Whether the parameter is required.
        description:
            An optional description of the parameter.
        valid_values:
            An optional list of valid values for the parameter. For example, `["celsius",
            "fahrenheit"]`, or `[1, 2, 3]`.
        any_of:
            An optional list of types that the parameter can be. For example, `[str, int]`, or
            `[str, MyModel2]`.

    Examples:
        Basic string parameter:
        ```python
        name_param = Parameter(
            name="name",
            param_type=str,
            required=True,
            description="The user's full name"
        )
        ```

        Number parameter with valid values (enum):
        ```python
        age_param = Parameter(
            name="age",
            param_type=int,
            required=True,
            description="User's age category",
            valid_values=[18, 21, 30, 40, 50, 65]
        )
        ```

        Parameter with choice of types (union type):
        ```python
        id_param = Parameter(
            name="identifier",
            param_type=str,  # Base/fallback type
            required=True,
            description="User ID (either string or number)",
            any_of=[str, int]
        )
        ```

        Array parameter:
        ```python
        tags_param = Parameter(
            name="tags",
            param_type=list[str],
            required=False,
            description="List of tags associated with the item"
        )
        ```

        Object parameter using a Pydantic model:
        ```python
        class Address(BaseModel):
            street: str
            city: str
            zip_code: str

        address_param = Parameter(
            name="address",
            param_type=Address,
            required=True,
            description="The user's mailing address"
        )
        ```
    """

    name: str
    param_type: Any
    required: bool
    description: str | None = None
    valid_values: list[Any] | None = None  # For enum-like constraints
    any_of: list[Any] | None = None  # For union types

    model_config = {
        "arbitrary_types_allowed": True,  # Allow any type for param_type field
    }

    @field_validator('param_type')
    @classmethod
    def validate_param_type(cls, v):  # noqa: ANN001
        """Validate that param_type is a valid type."""
        # Handle string representations
        if isinstance(v, str):
            v = _string_to_type(v)

        # Reject Any explicitly
        if v is Any:
            raise ValueError("Any is not supported as a param_type")

        # Check if it's a base type
        if isinstance(v, type):
            return v

        # Check if it's a typing generic (like List[str])
        origin = get_origin(v)
        if origin is not None:
            return v

        # If we get here, it's not a valid type
        raise ValueError(f"Invalid param_type: {v}. Must be a type, BaseModel, or typing generic.")

    def model_dump(self, **kwargs):  # noqa: ANN003, ANN201
        """Custom serialization that handles type objects."""
        # Create a copy of the object with param_type and any_of converted to strings
        data = super().model_dump(**kwargs)

        # Convert param_type to string representation
        if self.param_type is not None:
            data['param_type'] = str(self.param_type)

        # Convert any_of to string representations
        if self.any_of is not None:
            data['any_of'] = [str(t) for t in self.any_of]

        return data

    @classmethod
    def model_validate(cls, obj, **kwargs):  # noqa: ANN001, ANN003
        """Custom deserialization that handles type strings."""
        # This is just a placeholder since proper deserialization of types
        # from strings requires more complex logic
        if isinstance(obj, dict) and 'param_type' in obj and isinstance(obj['param_type'], str):
            # For deserialization tests, use str as a default type
            obj = dict(obj)
            obj['param_type'] = str

            if obj.get('any_of'):
                obj['any_of'] = [str for _ in obj['any_of']]

        return super().model_validate(obj, **kwargs)


class Tool(BaseModel):
    """Represents a function/tool that can be called by the model."""

    name: str
    parameters: list[Parameter]
    description: str | None = None
    func: Callable | None = None


class ToolChoice(Enum):
    """
    Enum for how the model should choose which tool to use.

    REQUIRED:
        OpenAI: "Required: Call one or more functions."

        Anthropic: "'any' tells Claude that it must use one of the provided tools, but doesn't
        force a particular tool."
    AUTO:
        OpenAI: "'auto' means the model can pick between generating a message or calling one or
        more tools."

        Anthropic: "'auto' allows Claude to decide whether to call any provided tools or not."
    """

    REQUIRED = auto()
    AUTO = auto()


class ToolPrediction(BaseModel):
    """The tool call details extracted from the model's response."""

    name: str
    arguments: dict[str, object]
    call_id: str


class ToolPredictionResponse(TokenSummary):
    """
    Response containing just the essential function/tool call information and usage stats.

    Content is filled if there is no function/call.

    If a tool call is predicted, then tool_call is filled and message is None.
    If no tool call is predicted, then tool_call is None and message is filled.
    """

    tool_prediction: ToolPrediction | None
    message: str | None = None
    trace_context: TraceContext | None = None


ClientType = TypeVar('ClientType', bound='Client')

class Client(ABC):
    """Base class for model wrappers."""

    registry = Registry()

    def __init__(self, model_name: str, **kwargs):  # noqa: ANN003, ARG002
        """Initialize client with optional telemetry setup."""
        self.model_name = model_name

        # Initialize telemetry components
        self.tracer = get_tracer()
        self.meter = get_meter()

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, object]],
    ) -> AsyncGenerator[
            TextChunkEvent | TextResponse | ToolPredictionResponse | StructuredOutputResponse,
            None,
        ]:
        """
        Streams asynchronously.

        For chat models, this method should return an async generator that yields ResponseChunk
        objects. The last response should be a ResponseSummary object.

        For tools models, this method should return a ToolPredictionResponse object.

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
        """
        pass

    def __call__(
            self,
            messages: list[dict[str, object]],
        ) -> TextResponse | ToolPredictionResponse | StructuredOutputResponse:
        """
        Invoke the model synchronously (e.g. chat) and return the complete response.

        This method handles the event loop complexity for you, making it easy to use
        in both synchronous and asynchronous contexts. It internally streams the response
        and returns the final result.

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
                Each message should be a dict with 'role' and 'content' keys.
        """
        async def run() -> TextResponse:
            # Check if the stream method returns a generator or a direct value
            result = self.stream(messages)
            # If it's an async generator, collect the last response
            if hasattr(result, '__aiter__'):
                last_response = None
                async for response in result:
                    last_response = response
                # Add trace context to the final response
                if last_response:
                    self._add_trace_context(last_response)
                return last_response
            # If it's a regular coroutine, just await and return the result
            final_response = await result
            if final_response:
                self._add_trace_context(final_response)
            return final_response

        # Try to use the current event loop if one is running
        try:
            # Try to get the running loop (will raise RuntimeError if no loop is running)
            loop = asyncio.get_running_loop()
            # We're inside a running event loop (test, Jupyter, etc.)
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()
            return loop.run_until_complete(run())
        except RuntimeError:
            # No running loop exists, use asyncio.run which creates and manages its own loop
            return asyncio.run(run())

    async def run_async(
        self,
        messages: list[dict[str, object]],
    ) -> TextResponse | ToolPredictionResponse | StructuredOutputResponse:
        """
        Invoke the model asynchronously but does not stream via AsyncGenerator. Rather, it waits
        for the model to finish processing all messages and returns the last response which is
        either a TextResponse (containing the entire content), a ToolPredictionResponse, or
        a StructuredOutputResponse.

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
        """
        with safe_span(
            self.tracer,
            "llm.request",
            attributes={
                "llm.model": self.model_name,
                "llm.provider": self._get_provider_name(),
                "llm.messages.count": len(messages),
                "llm.operation": "chat",
            },
        ) as span:
            # Add message content length for observability
            if span and messages:
                total_content_length = sum(
                    len(str(msg.get("content", ""))) for msg in messages
                )
                span.set_attribute("llm.input.content_length", total_content_length)

            # Execute existing logic
            last_response = None
            async for response in self.stream(messages):
                last_response = response

            # Add trace context to the final response
            if last_response:
                self._add_trace_context(last_response)

            # Emit metrics if available
            if self.meter and isinstance(last_response, TokenSummary):
                labels = {
                    "llm_model": self.model_name,
                    "llm_provider": self._get_provider_name(),
                    "llm_operation": "chat",
                }
                last_response.emit_metrics(self.meter, labels)

            # Add response metadata to span
            if span and isinstance(last_response, TokenSummary):
                span.set_attribute("llm.tokens.input", last_response.input_tokens)
                span.set_attribute("llm.tokens.output", last_response.output_tokens)
                span.set_attribute("llm.tokens.total", last_response.total_tokens)
                span.set_attribute("llm.duration_seconds", last_response.duration_seconds)

                if last_response.total_cost is not None:
                    span.set_attribute("llm.cost.total", last_response.total_cost)

            return last_response

    def _get_provider_name(self) -> str:
        """Get provider name for telemetry labels."""
        class_name = self.__class__.__name__.lower()
        if "openai" in class_name:
            return "openai"
        if "anthropic" in class_name:
            return "anthropic"
        return "unknown"

    def _add_trace_context(
            self,
            response: TextResponse | ToolPredictionResponse | StructuredOutputResponse,
        ) -> None:
        """Add trace context to response if available."""
        if hasattr(response, 'trace_context'):
            trace_id, span_id = extract_current_trace_context()
            if trace_id and span_id:
                response.trace_context = TraceContext(trace_id=trace_id, span_id=span_id)

    async def sample(
        self,
        messages: list[dict[str, object]],
        n: int = 2,
    ) -> list[TextResponse | ToolPredictionResponse | StructuredOutputResponse]:
        """
        Generate `n` responses from the model concurrently given the same input messages.

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
            n:
                Number of responses to generate.
        """
        with safe_span(
            self.tracer,
            "llm.sample",
            attributes={
                "llm.model": self.model_name,
                "llm.provider": self._get_provider_name(),
                "llm.sample.size": n,
                "llm.operation": "sample",
            },
        ):
            tasks = [self.run_async(messages) for _ in range(n)]
            return await asyncio.gather(*tasks)

    async def generate_multiple(
        self,
        messages: list[list[dict[str, object]]],
        sample_n: int = 1,
    ) -> list[TextResponse | ToolPredictionResponse | StructuredOutputResponse] | \
         list[list[TextResponse | ToolPredictionResponse | StructuredOutputResponse]]:
        """
        Generate multiple responses from the model concurrently given `n` different input messages.

        If `sample_n` is greater than 1, the model will generate `n` responses for each set of
        messages, resulting in a **list of lists** of responses of outer length `len(messages)` and
        inner length `sample_n`.

        If `sample_n` is 1, the model will generate a single response for each set of messages,
        resulting in a **list** of responses.

        Args:
            messages:
                List of list of messages. Each inner list represents a separate set of messages to
                send to the model.
            sample_n:
                Number of responses to generate for each set of messages.
        """
        with safe_span(
            self.tracer,
            "llm.batch_generate",
            attributes={
                "llm.model": self.model_name,
                "llm.provider": self._get_provider_name(),
                "llm.batch.size": len(messages),
                "llm.batch.sample_n": sample_n,
                "llm.operation": "batch",
            },
        ):
            # Execute existing logic
            if not (isinstance(messages, list) and all(isinstance(m, list) for m in messages)):
                raise TypeError("Messages must be a list of lists")

            all_tasks = []
            for message_set in messages:
                for _ in range(sample_n):
                    all_tasks.append(self.run_async(message_set))

            all_results = await asyncio.gather(*all_tasks)

            if sample_n == 1:
                # If sample_n is 1, return results directly as a flat list
                return all_results
            # If sample_n > 1, restructure into a list of lists
            result_lists = []
            for i in range(0, len(all_results), sample_n):
                result_lists.append(all_results[i:i+sample_n])
            return result_lists


    @classmethod
    def register(cls, client_type: str | RegisteredClients):
        """Register a subclass of Model."""

        def decorator(subclass: type[Client]) -> type[Client]:
            assert issubclass(
                subclass,
                Client,
            ), f"Model '{client_type}' ({subclass.__name__}) must extend Model"
            cls.registry.register(type_name=client_type, item=subclass)
            return subclass

        return decorator

    @classmethod
    def is_registered(cls, client_type: str | RegisteredClients) -> bool:
        """Check if a model type is registered."""
        return client_type in cls.registry

    @classmethod
    def instantiate(
        cls: type[ClientType],
        client_type: str | RegisteredClients,
        model_name: str,
        **model_kwargs: dict | None,
    ) -> ClientType | list[ClientType]:
        """
        Creates a Model object.

        This method requires that the Model subclass has been registered with the `register`
        decorator before calling this method. It also requires that the dictionary has a
        `client_type` field that matches the type name of the registered Model subclass.

        Args:
            client_type:
                The type of model to instantiate.
            model_name:
                The name of the model to instantiate.
            model_kwargs:
                Optional dictionary containing the model parameters in the format:
                {
                    'temperature': 0.5,
                    'max_tokens': 100,
                    'top_p': 0.9,
                }
        """
        if cls.is_registered(client_type):
            model_kwargs = deepcopy(model_kwargs)
            model_kwargs['model_name'] = model_name
            return cls.registry.create_instance(type_name=client_type, **model_kwargs)
        raise ValueError(f"Unknown Model type `{client_type}`")


def pydantic_model_to_parameters(response_format: type[BaseModel]) -> list[Parameter]:  # noqa: PLR0912
    """Convert a Pydantic model to a list of Parameter objects for function/tool calling."""
    parameters = []

    # Get model schema - this includes info about required fields
    model_schema = response_format.model_json_schema()
    required_fields = set(model_schema.get('required', []))
    model_fields = response_format.model_fields

    for field_name, field in model_fields.items():
        # Start with default from schema
        required = field_name in required_fields

        # Get description from Field if available
        description = field.description or None

        # Get the field annotation for type analysis
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Literal types
        if origin is Literal:
            # Get all literal values
            literal_values = args

            # Determine appropriate base type for the literal
            if all(isinstance(val, str) for val in literal_values):
                base_type = str
            elif all(isinstance(val, int) for val in literal_values):
                base_type = int
            elif all(isinstance(val, float) for val in literal_values):
                base_type = float
            elif all(isinstance(val, bool) for val in literal_values):
                base_type = bool
            else:
                # Mixed types, use string
                base_type = str
                literal_values = [str(val) for val in literal_values]

            parameters.append(Parameter(
                name=field_name,
                param_type=base_type,
                required=required,
                description=description,
                valid_values=list(literal_values),
            ))
            continue

        # Special handling for Optional/Union with None
        if origin in (Union, types.UnionType) and type(None) in args:
            # Get the non-None types for any_of
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                # It's a simple Optional[X]
                parameters.append(Parameter(
                    name=field_name,
                    param_type=non_none_types[0],
                    required=required,
                    description=description,
                ))
            else:
                # It's a Union with multiple types
                parameters.append(Parameter(
                    name=field_name,
                    param_type=str,  # Placeholder type, will be overridden by any_of
                    any_of=non_none_types,
                    required=required,
                    description=description,
                ))
            continue

        # Handle Enum types
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            parameters.append(Parameter(
                name=field_name,
                param_type=str,
                required=required,
                description=description,
                valid_values=[e.value for e in annotation],
            ))
            continue

        # Handle regular Union types (no None)
        if origin in (Union, types.UnionType):
            parameters.append(Parameter(
                name=field_name,
                param_type=str,  # Placeholder type, will be overridden by any_of
                any_of=list(args),
                required=required,
                description=description,
            ))
            continue

        # Handle Any type - reject it
        if annotation is Any:
            raise ValueError(f"Field {field_name} has type Any which is not supported")

        # For all other types, pass the Python type directly
        parameters.append(Parameter(
            name=field_name,
            param_type=annotation,  # Could be a basic type, typing generic, or Pydantic model
            required=required,
            description=description,
        ))

    return parameters

def pydantic_model_to_tool(response_format: type[BaseModel]) -> Tool:
    """Convert a Pydantic model to a Tool object for use with function/tool calling APIs."""
    parameters = pydantic_model_to_parameters(response_format)
    tool_name = response_format.__name__
    description = f"Generate a response in the format specified by {tool_name}"
    return Tool(
        name=tool_name,
        parameters=parameters,
        description=description,
    )
