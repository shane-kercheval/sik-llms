"""Base classes and utilities for models."""
import asyncio
import inspect
import json
import types
import nest_asyncio
from pydantic import BaseModel
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from copy import deepcopy
from enum import Enum, auto
from typing import List, Literal, TypeVar, Union, get_args, get_origin  # noqa: UP035
from sik_llms.utilities import Registry, get_json_schema_type


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


class ContentType(Enum):
    """Enum for content chunk types."""

    TEXT = auto()
    THINKING = auto()
    TOOL_PREDICTION = auto()
    TOOL_RESULT = auto()
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


class ResponseChunk(BaseModel):
    """A chunk returned when streaming."""

    content: str | object
    content_type: ContentType = ContentType.TEXT
    logprob: float | None = None
    iteration: int | None = None


class StructuredOutputResponse(BaseModel):
    """Response containing structured output data."""

    parsed: BaseModel | None
    refusal: str | object | None


class TokenSummary(BaseModel):
    """Summary of a chat response."""

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    duration_seconds: float

    @property
    def total_tokens(self) -> int:
        """Calculate the total number of tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        """Calculate the total cost."""
        return self.input_cost + self.output_cost


class ResponseSummary(TokenSummary):
    """Summary of a chat response."""

    content: str | StructuredOutputResponse


class Parameter(BaseModel):
    """Represents a parameter property in a Tools's schema."""

    name: str
    type: Literal['string', 'number', 'boolean', 'integer', 'object', 'array', 'enum', 'anyOf']
    required: bool
    description: str | None = None
    enum: list[str | int | float | bool] | None = None


class Tool(BaseModel):
    """Represents a function/tool that can be called by the model."""

    name: str
    parameters: list[Parameter]
    description: str | None = None

    def to_openai(self) -> dict[str, object]:
        """Convert the tool to the format expected by OpenAI API."""
        properties = {}
        required = []
        for param in self.parameters:
            param_dict = {'type': param.type}
            if param.description:
                param_dict['description'] = param.description
            if param.enum:
                param_dict['enum'] = param.enum
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

        # If all properties are required we should use `strict` mode
        # However, "All fields in properties must be marked as required" (in order to use strict)
        # https://platform.openai.com/docs/guides/function-calling
        # so we should set `strict` to True only if all parameters are required
        strict = all(param.required for param in self.parameters or [])
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'strict': strict,
                **({'description': self.description} if self.description else {}),
                'parameters': parameters_dict,
            },
        }

    def to_anthropic(self) -> dict[str, object]:
        """Convert the tool to the format expected by OpenAI API."""
        properties = {}
        required = []
        for param in self.parameters:
            param_dict = {'type': param.type}
            if param.description:
                param_dict['description'] = param.description
            if param.enum:
                param_dict['enum'] = param.enum
            properties[param.name] = param_dict
            if param.required:
                required.append(param.name)

        parameters_dict = {
            'type': 'object',
            'properties': properties,
        }
        if required:
            parameters_dict['required'] = required

        return {
            'name': self.name,
            **({'description': self.description} if self.description else {}),
            'input_schema': parameters_dict,
        }


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


ClientType = TypeVar('ClientType', bound='Client')

class Client(ABC):
    """Base class for model wrappers."""

    registry = Registry()

    def __call__(
            self,
            messages: list[dict[str, object]],
        ) -> ResponseSummary | ToolPredictionResponse:
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
        async def run() -> ResponseSummary:
            # Check if the run_async method returns a generator or a direct value
            result = self.run_async(messages)
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
    ) -> AsyncGenerator[ResponseChunk | ResponseSummary, None] | ToolPredictionResponse:
        """
        Run asynchronously.

        For chat models, this method should return an async generator that yields ResponseChunk
        objects. The last response should be a ResponseSummary object.

        For tools models, this method should return a ToolPredictionResponse object.

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
        cls: type[ClientType],
        client_type: str | Enum,
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


def pydantic_model_to_parameters(model_class: BaseModel) -> list[Parameter]:
    """Convert a Pydantic model to a list of Parameter objects for function/tool calling."""
    parameters = []

    # Get model schema - this includes info about required fields
    model_schema = model_class.model_json_schema()
    required_fields = set(model_schema.get('required', []))
    model_fields = model_class.model_fields

    for field_name, field in model_fields.items():
        # Start with default from schema
        required = field_name in required_fields

        # Get the field annotation for type analysis
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        # Special handling for Optional/Union with None
        # In Pydantic v2, Optional fields may still be listed as required in the schema
        # if they don't have default values, so we need to check the type directly
        if origin in (Union, types.UnionType) and type(None) in args:
            # Get the non-None type for schema generation
            non_none_type = next(arg for arg in args if arg is not type(None))
            annotation = non_none_type

        # Get description from Field if available
        description = field.description or None
        # Handle Enum types specifically
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            parameters.append(Parameter(
                name=field_name,
                type="enum",
                required=required,
                description=description,
                enum=[e.value for e in annotation],
            ))
            continue

        # Handle List[Model] types
        if origin in (list, List) and args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):  # noqa: E501, UP006
            nested_model = args[0]
            nested_schema = nested_model.model_json_schema()
            model_desc = f"An array of {nested_model.__name__} objects with the following structure:\n{json.dumps(nested_schema, indent=2)}"  # noqa: E501

            parameters.append(Parameter(
                name=field_name,
                type="array",
                required=required,
                description=model_desc,
            ))
            continue

        # Handle nested Pydantic models
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            # Include both the field description AND the model schema
            nested_schema = annotation.model_json_schema()
            model_desc = f"A {annotation.__name__} object with the following structure:\n{json.dumps(nested_schema, indent=2)}"  # noqa: E501

            full_description = description or ""
            if full_description:
                full_description += "\n\n"
            full_description += model_desc

            parameters.append(Parameter(
                name=field_name,
                type="object",
                required=required,
                description=full_description,
            ))
            continue

        # For regular types
        json_type, extra_props = get_json_schema_type(annotation)

        # Include both the field description AND any extra properties
        full_description = description or ""
        if extra_props and extra_props != {}:
            if full_description:
                full_description += "\n\n"
            full_description += f"Additional schema properties: {json.dumps(extra_props, indent=2)}"  # noqa: E501

        parameters.append(Parameter(
            name=field_name,
            type=json_type,
            required=required,
            description=full_description or None,
        ))
    return parameters


def pydantic_model_to_tool(model_class: BaseModel) -> Tool:
    """Convert a Pydantic model to a Tool object for use with function/tool calling APIs."""
    parameters = pydantic_model_to_parameters(model_class)
    tool_name = model_class.__name__
    description = f"Generate a response in the format specified by {tool_name}"
    return Tool(
        name=tool_name,
        parameters=parameters,
        description=description,
    )
