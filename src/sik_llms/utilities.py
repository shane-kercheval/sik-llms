"""Utility functions and classes."""
from enum import Enum
from typing import TypeVar


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
