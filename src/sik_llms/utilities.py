"""Utility functions and classes."""
from enum import Enum
import inspect
import types
from typing import TypeVar, get_origin, get_args, Any, Dict, List, Union  # noqa: UP035
from pydantic import BaseModel


T = TypeVar('T')

class Registry:
    """
    A registry for managing different types of classes.
    Allows for registering classes with a type name and creating instances of these classes.
    The registry is case-insensitive for type names.
    """

    def __init__(self):
        """Initialize the registry with an empty dictionary."""
        self._registry: dict[str, type[T]] = {}

    @staticmethod
    def _clean_type_name(type_name: str | Enum) -> str:
        """Convert the type name to uppercase."""
        if isinstance(type_name, Enum):
            return type_name.name.upper()
        return type_name.upper()

    def register(self, type_name: str | Enum, item: type[T]) -> None:
        """
        Register a class with a specified type name (case-insensitive).

        If the type name is already registered, an assertion error is raised.

        Args:
            type_name:
                The type name to be associated with the class.

                The type_name is case-insensitive. If the type_name is an Enum, the name
                (`type_name.name`) is used.
            item: The class to be registered.
        """
        type_name = self._clean_type_name(type_name)
        assert type_name not in self._registry, f"Type '{type_name}' already registered."
        item._type_name = type_name
        self._registry[type_name] = item

    def get(self, type_name: str | Enum) -> type[T]:
        """
        Get the class associated with the given type name.

        Args:
            type_name: The type name of the class to retrieve.
        """
        type_name = self._clean_type_name(type_name)
        return self._registry[type_name]

    def __contains__(self, type_name: str | Enum) -> bool:
        """
        Check if a type name is registered in the registry (case insensitive).

        Args:
            type_name: The type name to check.
        """
        type_name = self._clean_type_name(type_name)
        return type_name in self._registry

    def create_instance(self, type_name: str | Enum, **data: dict) -> T:
        """
        Create an instance of the class associated with the given type name.

        Args:
            type_name (str): The type name of the class to instantiate.
            data: Keyword arguments to be passed to the class constructor.

        Raises:
            ValueError: If the type name is not registered in the registry.
        """
        if isinstance(type_name, Enum):
            type_name = type_name.name
        if type_name.upper() not in self._registry:
            raise ValueError(f"Unknown type `{type_name}`")
        return self._registry[type_name.upper()](**data)


def get_json_schema_type(annotation: type) -> tuple[str, dict[str, Any]]:  # noqa: PLR0911, PLR0912
    """
    Convert a Python type annotation to a JSON Schema type.

    Returns:
        Tuple of (type_name, extra_properties)

    Raises:
        ValueError: For unsupported type annotations
    """
    # Handle None/Optional types
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Any type explicitly
    if annotation is Any:
        raise ValueError(f"Unsupported type annotation: {annotation}")

    # Handle object type explicitly
    if annotation is object:
        raise ValueError(f"Unsupported type annotation: {annotation}")

    # Handle Union types (including Optional)
    if origin is Union or origin is types.UnionType:
        # Check if None or NoneType is in the union
        if type(None) in args:
            # Get the other type(s) in the Union
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                # It's an Optional[X] / X | None
                return get_json_schema_type(non_none_args[0])

        # For unions with multiple types that aren't Optional
        any_of_schemas = []
        for arg in args:
            if arg is not type(None):
                type_name, props = get_json_schema_type(arg)
                any_of_schemas.append({"type": type_name, **props})
        return "anyOf", {"anyOf": any_of_schemas}

    # Handle primitive types
    if annotation is str or annotation is str:
        return "string", {}
    if annotation is int or annotation is int:
        return "integer", {}
    if annotation is float or annotation is float:
        return "number", {}
    if annotation is bool or annotation is bool:
        return "boolean", {}

    # Handle lists/arrays
    if origin is list or origin is List or annotation is list:  # noqa: UP006
        if args:
            item_type, item_props = get_json_schema_type(args[0])
            return "array", {"items": {"type": item_type, **item_props}}
        return "array", {"items": {"type": "string"}}  # Default to string items

    # Handle tuples as arrays (with tuple validation if needed)
    if origin is tuple or annotation is tuple:
        if not args:
            # Plain tuple with no type args
            return "array", {"items": {"type": "string"}}

        # Handle homogeneous tuples (Tuple[X, ...])
        if len(args) == 2 and args[1] is Ellipsis:
            item_type, item_props = get_json_schema_type(args[0])
            return "array", {"items": {"type": item_type, **item_props}}

        # Handle heterogeneous tuples (Tuple[X, Y, Z])
        items = []
        for arg in args:
            item_type, item_props = get_json_schema_type(arg)
            items.append({"type": item_type, **item_props})

        return "array", {
            "prefixItems": items,
            "minItems": len(items),
            "maxItems": len(items),
        }

    # Handle sets as arrays
    if origin is set or annotation is set:
        if args:
            item_type, item_props = get_json_schema_type(args[0])
            return "array", {
                "items": {"type": item_type, **item_props},
                "uniqueItems": True,
            }
        return "array", {"items": {"type": "string"}, "uniqueItems": True}

    # Handle dictionaries/objects
    if origin is dict or origin is Dict or annotation is dict:  # noqa: UP006
        if len(args) >= 2:  # noqa: SIM102
            if args[0] is str or args[0] is str:
                value_type, value_props = get_json_schema_type(args[1])
                return "object", {"additionalProperties": {"type": value_type, **value_props}}
        return "object", {"additionalProperties": True}

    # Handle Enum types
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return "string", {"enum": [e.value for e in annotation]}

    # Handle nested Pydantic models
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        # For nested models, we return an object type with a reference
        return "object", {
            "type": "object",
            "description": f"A {annotation.__name__} object",
        }

    # Raise ValueError for unhandled types
    raise ValueError(f"Unsupported type annotation: {annotation}")
