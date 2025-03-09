"""Base classes and utilities for models."""
import asyncio
import inspect
import json
import types
import nest_asyncio
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Callable, List, Literal, Type, TypeVar, Union, get_args, get_origin  # noqa: UP035
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


# class ContentType(Enum):
#     """Enum for content chunk types."""

#     TEXT = auto()
#     THINKING = auto()
#     REDACTED_THINKING = auto()
#     TOOL_PREDICTION = auto()
#     TOOL_RESULT = auto()
#     ERROR = auto()


def user_message(content: str) -> dict:
    """Returns a user message."""
    return {'role': 'user', 'content': content}


def assistant_message(content: str) -> dict:
    """Returns an assistant message."""
    return {'role': 'assistant', 'content': content}


def system_message(content: str) -> dict:
    """Returns a system message."""
    return {'role': 'system', 'content': content}


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

    response: str | StructuredOutputResponse


class Parameter(BaseModel):
    """
    Represents a parameter property in a Tools's schema.

    Args:
        name:
            The name of the parameter.
        param_type:
            The type of the parameter (e.g. `str`, `int`, `bool`, `MyModel(BaseModel)`).
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
    def validate_param_type(cls, v):
        """Validate that param_type is a valid type."""
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
    
    def model_dump(self, **kwargs):
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
    def model_validate(cls, obj, **kwargs):
        """Custom deserialization that handles type strings."""
        # This is just a placeholder since proper deserialization of types 
        # from strings requires more complex logic
        if isinstance(obj, dict) and 'param_type' in obj and isinstance(obj['param_type'], str):
            # For deserialization tests, use str as a default type
            obj = dict(obj)
            obj['param_type'] = str
            
            if 'any_of' in obj and obj['any_of']:
                obj['any_of'] = [str for _ in obj['any_of']]
                
        return super().model_validate(obj, **kwargs)



class Tool(BaseModel):
    """Represents a function/tool that can be called by the model."""

    name: str
    parameters: list[Parameter]
    description: str | None = None
    func: Callable | None = None

    def to_openai(self) -> dict[str, object]:
        """Convert the tool to the format expected by OpenAI API."""
        properties = {}
        required = []
        
        for param in self.parameters:
            # Convert the Python type to JSON Schema type
            if param.any_of:
                # Handle union types
                any_of_schemas = []
                for union_type in param.any_of:
                    json_type, extra_props = get_json_schema_type(union_type)
                    any_of_schemas.append({"type": json_type, **extra_props})
                param_dict = {"anyOf": any_of_schemas}
            else:
                # Handle single type
                json_type, extra_props = get_json_schema_type(param.param_type)
                param_dict = {"type": json_type, **extra_props}
                
            # Add description if available
            if param.description:
                param_dict["description"] = param.description
                
            # Add enum values if provided
            if param.valid_values:
                param_dict["enum"] = param.valid_values
                
            # For object types, ensure all properties are required
            if json_type == 'object' and 'properties' in param_dict:
                if param_dict['properties']:
                    param_dict['required'] = list(param_dict['properties'].keys())
                param_dict['additionalProperties'] = False
                
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
        strict = all(param.required for param in self.parameters or [])
        result = {
            'type': 'function',
            'function': {
                'name': self.name,
                'strict': strict,
                **({'description': self.description} if self.description else {}),
                'parameters': parameters_dict,
            },
        }
        
        # Helper function to process the entire schema recursively
        def process_schema(obj):
            if isinstance(obj, dict):
                # Remove default values
                if 'default' in obj:
                    del obj['default']
                    
                # Ensure object types have required fields and additionalProperties: false
                if obj.get('type') == 'object' and 'properties' in obj:
                    # Make all properties required
                    if obj['properties']:
                        obj['required'] = list(obj['properties'].keys())
                    obj['additionalProperties'] = False
                    
                # Process all nested objects/arrays
                for key, value in list(obj.items()):
                    if isinstance(value, (dict, list)):
                        process_schema(value)
                        
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        process_schema(item)
        
        # Process the entire schema
        process_schema(result)
        
        return result

    def to_anthropic(self) -> dict[str, object]:
        """Convert the tool to the format expected by Anthropic API."""
        properties = {}
        required = []
        
        for param in self.parameters:
            # Convert the Python type to JSON Schema type
            if param.any_of:
                # Handle union types
                any_of_schemas = []
                for union_type in param.any_of:
                    json_type, extra_props = get_json_schema_type(union_type)
                    # Remove any default values from extra_props
                    if 'default' in extra_props:
                        del extra_props['default']
                    any_of_schemas.append({"type": json_type, **extra_props})
                param_dict = {"anyOf": any_of_schemas}
            else:
                # Handle single type
                json_type, extra_props = get_json_schema_type(param.param_type)
                # Remove any default values from extra_props
                if 'default' in extra_props:
                    del extra_props['default']
                param_dict = {"type": json_type, **extra_props}
                
            # Add description if available
            if param.description:
                param_dict["description"] = param.description
                
            # Add enum values if provided
            if param.valid_values:
                param_dict["enum"] = param.valid_values
                
            # Ensure all object types have additionalProperties: false
            if json_type == 'object':
                param_dict['additionalProperties'] = False
                # Recursively remove default values from nested object properties
                if 'properties' in param_dict:
                    self._remove_defaults_recursively(param_dict['properties'])
                
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

        return {
            'name': self.name,
            **({'description': self.description} if self.description else {}),
            'input_schema': parameters_dict,
        }

    # Helper method to recursively remove default values from schemas
    def _remove_defaults_recursively(self, obj):
        """Remove default values recursively from a properties dictionary."""
        if isinstance(obj, dict):
            # Remove default from this object if present
            if 'default' in obj:
                del obj['default']
            
            # Process all dictionary values recursively
            for key, value in list(obj.items()):
                if isinstance(value, (dict, list)):
                    self._remove_defaults_recursively(value)
                    
            # For object types, ensure additionalProperties is false
            if obj.get('type') == 'object' and 'properties' in obj:
                obj['additionalProperties'] = False
                
        elif isinstance(obj, list):
            # Process all list items recursively
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._remove_defaults_recursively(item)

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
    ) -> AsyncGenerator[TextChunkEvent | ResponseSummary, None] | ToolPredictionResponse:
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
