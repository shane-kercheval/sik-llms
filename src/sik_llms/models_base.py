"""Base classes and utilities for models."""
import asyncio
import nest_asyncio
from pydantic import BaseModel
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from copy import deepcopy
from enum import Enum, auto
from typing import TypeVar
from sik_llms.utilities import Registry


class RegisteredClients(Enum):
    """Enum for model types."""

    OPENAI = 'OpenAI'
    OPENAI_FUNCTIONS = 'OpenAIFunctions'
    ANTHROPIC = 'Anthropic'


class ReasoningEffort(Enum):
    """Enum for reasoning effort levels."""

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class ContentType(Enum):
    """Enum for content chunk types."""

    TEXT = auto()
    THINKING = auto()
    REDACTED_THINKING = auto()
    ERROR = auto()


def user_message(content: str) -> dict:
    """Returns a user message."""
    return {'role': 'user', 'content': content}


def assistant_message(content: str) -> dict:
    """Returns an assistant message."""
    return {'role': 'assistant', 'content': content}


def system_message(content: str) -> dict:
    """Returns a system message."""
    return {'role': 'system', 'content': content}


class ChatChunkResponse(BaseModel):
    """A chunk returned when streaming."""

    content: str | object
    content_type: ContentType = ContentType.TEXT
    logprob: float | None = None


class StructuredOutputResponse(BaseModel):
    """Response containing structured output data."""

    parsed: BaseModel | None
    refusal: str | object | None


class ChatResponseSummary(BaseModel):
    """Summary of a chat response."""

    content: str | StructuredOutputResponse
    total_input_tokens: int
    total_output_tokens: int
    total_input_cost: float
    total_output_cost: float
    duration_seconds: float


class Parameter(BaseModel):
    """
    Represents a parameter property in a function's schema.

    Supported types
        The following types are supported for Structured Outputs:

        String
        Number
        Boolean
        Integer
        Object
        Array
        Enum
        anyOf

    """

    name: str
    type: str
    required: bool
    description: str | None = None
    enum: list[str] | None = None


class Function(BaseModel):
    """Represents a function that can be called by the model."""

    name: str
    parameters: list[Parameter]
    description: str | None = None

    def to_openai_schema(self) -> dict[str, object]:
        """Convert the function to the format expected by OpenAI API."""
        properties = {}
        required = []

        for param in self.parameters:
            param_dict = {"type": param.type}
            if param.description:
                param_dict["description"] = param.description
            if param.enum:
                param_dict["enum"] = param.enum
            properties[param.name] = param_dict
            if param.required:
                required.append(param.name)

        parameters_dict = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters_dict["required"] = required
        parameters_dict["additionalProperties"] = False

        # If all properties are required we should use `strict` mode
        # However, "All fields in properties must be marked as required" (in order to use strict)
        # https://platform.openai.com/docs/guides/function-calling
        # so we should set `strict` to True only if all parameters are required
        strict = all(param.required for param in self.parameters or [])
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "strict": strict,
                **({"description": self.description} if self.description else {}),
                "parameters": parameters_dict,
            },
        }


class FunctionCallResult(BaseModel):
    """The function call details extracted from the model's response."""

    name: str
    arguments: dict[str, object]
    call_id: str


class FunctionCallResponse(BaseModel):
    """Response containing just the essential function call information and usage stats."""

    function_call: FunctionCallResult
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float


M = TypeVar('M', bound='Client')


class Client(ABC):
    """Base class for model wrappers."""

    registry = Registry()

    def __call__(
        self,
        messages: list[dict[str, object]],
        model_name: str | None = None,
        **model_kwargs: dict[str, object],
    ) -> ChatResponseSummary | FunctionCallResponse:
        """
        Invoke the model (e.g. chat).

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            **model_kwargs:
                Additional parameters to pass to the API call (e.g. temperature, max_tokens).
        """
        async def run() -> ChatResponseSummary:
            # Check if the run_async method returns a generator or a direct value
            result = self.run_async(messages, model_name=model_name, **model_kwargs)
            # If it's an async generator, collect the last response
            if hasattr(result, '__aiter__'):
                last_response = None
                async for response in result:
                    last_response = response
                return last_response
            # If it's a regular coroutine, just await and return the result
            return await result

        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
            # Check if the loop is already running
            if loop.is_running():
                # We're inside a running event loop (test, Jupyter, etc.)
                # Apply nest_asyncio to allow nested event loops
                nest_asyncio.apply()
                return loop.run_until_complete(run())
            # We have a loop but it's not running
            return loop.run_until_complete(run())
        except RuntimeError:
            # No event loop exists
            return asyncio.run(run())

    @abstractmethod
    async def run_async(
        self,
        messages: list[dict[str, object]],
        model_name: str | None = None,
        **model_kwargs: dict[str, object],
    ) -> AsyncGenerator[ChatChunkResponse | ChatResponseSummary, None] | FunctionCallResponse:
        """
        Run asynchronously.

        For chat models, this method should return an async generator that yields ChatChunkResponse
        objects. The last response should be a ChatResponseSummary object.

        For function models, this method should return a FunctionCallResponse object.

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            **model_kwargs:
                Additional parameters to pass to the API call (e.g. temperature, max_tokens).
        """
        pass

    @classmethod
    def register(cls, client_type: str | Enum):
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
    def is_registered(cls, client_type: str | Enum) -> bool:
        """Check if a model type is registered."""
        return client_type in cls.registry

    @classmethod
    def instantiate(
        cls: type[M],
        client_type: str | Enum,
        model_name: str,
        **model_kwargs: dict | None,
    ) -> M | list[M]:
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
