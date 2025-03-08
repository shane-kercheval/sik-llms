"""Tests for the utilities module."""
import pytest
from enum import Enum, auto
from typing import List, Dict, Optional, Union, Any  # noqa: UP035
from pydantic import BaseModel
from sik_llms.utilities import Registry
from sik_llms.utilities import get_json_schema_type


class SampleEnum(Enum):
    """Enum for testing Registry."""

    TYPE1 = auto()
    TYPE2 = auto()

class SampleClass1:
    """Sample class for testing Registry."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class SampleClass2:
    """Sample class for testing Registry."""

    def __init__(self, name: str):
        self.name = name

def test__Registry__register_and_get_with_string():
    """Test registering and retrieving a class using a string type name."""
    registry = Registry()
    registry.register("sample_class_1", SampleClass1)
    assert "sample_class_1" in registry
    assert "SAMPLE_CLASS_1" in registry
    assert registry.get("sample_class_1") == SampleClass1
    assert registry.get("SAMPLE_CLASS_1") == SampleClass1

def test__Registry__register_and_get_with_enum():
    """Test registering and retrieving a class using an Enum type name."""
    registry = Registry()
    registry.register(SampleEnum.TYPE1, SampleClass1)
    assert SampleEnum.TYPE1 in registry
    assert "TYPE1" in registry
    assert "type1" in registry
    assert registry.get("TYPE1") == SampleClass1
    assert registry.get(SampleEnum.TYPE1) == SampleClass1

def test__Registry__duplicate_registration_raises_error():
    """Test that registering a type name twice raises an assertion error."""
    registry = Registry()
    registry.register("duplicate", SampleClass1)
    with pytest.raises(AssertionError, match="Type 'DUPLICATE' already registered."):
        registry.register("duplicate", SampleClass2)

def test__Registry__create_instance_with_string():
    registry = Registry()
    """Test creating an instance using a string type name."""
    registry.register("sample_class_1", SampleClass1)
    instance = registry.create_instance("sample_class_1", x=10, y=20)
    assert isinstance(instance, SampleClass1)
    assert instance.x == 10
    assert instance.y == 20

    registry.register("sample_class_2", SampleClass2)
    instance = registry.create_instance("sample_class_2", name="test")
    assert isinstance(instance, SampleClass2)
    assert instance.name == "test"

def test__Registry__create_instance_with_enum():
    """Test creating an instance using an Enum type name."""
    registry = Registry()
    registry.register(SampleEnum.TYPE1, SampleClass1)
    instance = registry.create_instance(SampleEnum.TYPE1, x=30, y=40)
    assert isinstance(instance, SampleClass1)
    assert instance.x == 30
    assert instance.y == 40

    instance = registry.create_instance("TYPE1", x=50, y=60)
    assert isinstance(instance, SampleClass1)
    assert instance.x == 50
    assert instance.y == 60

    registry.register(SampleEnum.TYPE2, SampleClass2)
    instance = registry.create_instance(SampleEnum.TYPE2, name="test")
    assert isinstance(instance, SampleClass2)
    assert instance.name == "test"

def test__Registry__create_instance_unregistered_type_raises_error():
    registry = Registry()
    """Test that trying to create an instance of an unregistered type raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown type `sample_class_2`"):
        registry.create_instance("sample_class_2", name="test")

def test__Registry__clean_type_name_with_string():
    """Test the _clean_type_name method with a string input."""
    assert Registry._clean_type_name("sample") == "SAMPLE"

def test__Registry__clean_type_name_with_enum():
    """Test the _clean_type_name method with an Enum input."""
    assert Registry._clean_type_name(SampleEnum.TYPE1) == "TYPE1"


# Define test classes
class TestEnum(Enum):  # noqa: D101
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Address(BaseModel):  # noqa: D101
    street: str
    city: str
    zip_code: str


class Person(BaseModel):  # noqa: D101
    name: str
    age: int
    address: Address


class Test_get_json_schema_type:  # noqa: D101, N801

    def test_primitive_types(self):
        """Test conversion of primitive Python types to JSON Schema types."""
        assert get_json_schema_type(str) == ('string', {})
        assert get_json_schema_type(int) == ('integer', {})
        assert get_json_schema_type(float) == ('number', {})
        assert get_json_schema_type(bool) == ('boolean', {})


    def test_list_types(self):
        """Test conversion of list/array types to JSON Schema types."""
        # Simple list of strings
        type_name, props = get_json_schema_type(List[str])  # noqa: UP006
        assert type_name == 'array'
        assert props == {'items': {'type': 'string'}}

        type_name_alt, props_alt = get_json_schema_type(list[str])
        assert type_name_alt == type_name
        assert props_alt == props

        # List of integers
        type_name, props = get_json_schema_type(List[int])  # noqa: UP006
        assert type_name == 'array'
        assert props == {'items': {'type': 'integer'}}

        type_name_alt, props_alt = get_json_schema_type(list[int])
        assert type_name_alt == type_name
        assert props_alt == props

        # Plain list (no type args)
        type_name, props = get_json_schema_type(List)  # noqa: UP006
        assert type_name == 'array'
        assert props['items']['type'] == 'string'  # Default item type

        type_name_alt, props_alt = get_json_schema_type(list)
        assert type_name_alt == type_name
        assert props_alt == props


    def test_dict_types(self):
        """Test conversion of dict/object types to JSON Schema types."""
        # Dict with string keys and integer values
        type_name, props = get_json_schema_type(Dict[str, int])  # noqa: UP006
        assert type_name == 'object'
        assert props['additionalProperties']['type'] == 'integer'

        type_name_alt, props_alt = get_json_schema_type(dict[str, int])
        assert type_name_alt == type_name
        assert props_alt == props

        # Dict with string keys and string values
        type_name, props = get_json_schema_type(Dict[str, str])  # noqa: UP006
        assert type_name == 'object'
        assert props['additionalProperties']['type'] == 'string'

        type_name_alt, props_alt = get_json_schema_type(dict[str, str])
        assert type_name_alt == type_name
        assert props_alt == props

        # Plain dict (no type args)
        type_name, props = get_json_schema_type(Dict)  # noqa: UP006
        assert type_name == 'object'
        assert props['additionalProperties'] is True

        type_name_alt, props_alt = get_json_schema_type(dict)
        assert type_name_alt == type_name
        assert props_alt == props


    def test_optional_types(self):
        """Test conversion of Optional types to JSON Schema types."""
        # Optional string (Union[str, None])
        type_name, props = get_json_schema_type(Optional[str])
        assert type_name == 'string'

        # Optional int
        type_name, props = get_json_schema_type(Optional[int])
        assert type_name == 'integer'

        # Optional list of strings
        type_name, props = get_json_schema_type(Optional[List[str]])  # noqa: UP006
        assert type_name == 'array'
        assert props['items']['type'] == 'string'

        class TestModel(BaseModel):
            param: str | None = None
            param_2: list[str] | None = None

        annotation = dict(TestModel.__annotations__.items())['param']
        type_name, props = get_json_schema_type(annotation)
        assert type_name == 'string'
        assert props == {}

        annotation = dict(TestModel.__annotations__.items())['param_2']
        type_name, props = get_json_schema_type(annotation)
        assert type_name == 'array'
        assert props['items']['type'] == 'string'


    def test_union_types(self):
        """Test conversion of Union types to JSON Schema types."""
        # Union of string and int
        type_name, props = get_json_schema_type(Union[str, int])
        assert type_name == 'anyOf'
        assert any(schema['type'] == 'string' for schema in props['anyOf'])
        assert any(schema['type'] == 'integer' for schema in props['anyOf'])

        # Union with None (should be handled as Optional)
        type_name, props = get_json_schema_type(Union[str, None])
        assert type_name == 'string'

        class TestModel(BaseModel):
            param: str | None = None
            param_2: list[str] | None = None
            param_3: str | int

        annotation = dict(TestModel.__annotations__.items())['param']
        type_name, props = get_json_schema_type(annotation)
        assert type_name == 'string'
        assert props == {}

        annotation = dict(TestModel.__annotations__.items())['param_2']
        type_name, props = get_json_schema_type(annotation)
        assert type_name == 'array'
        assert props['items']['type'] == 'string'

        annotation = dict(TestModel.__annotations__.items())['param_3']
        type_name, props = get_json_schema_type(annotation)
        assert type_name == 'anyOf'
        assert any(schema['type'] == 'string' for schema in props['anyOf'])
        assert any(schema['type'] == 'integer' for schema in props['anyOf'])


    def test_nested_container_types(self):
        """Test conversion of nested container types to JSON Schema types."""
        # List of lists of strings
        type_name, props = get_json_schema_type(List[List[str]])  # noqa: UP006
        assert type_name == 'array'
        assert props['items']['type'] == 'array'
        assert props['items']['items']['type'] == 'string'

        type_name_alt, props_alt = get_json_schema_type(list[list[str]])
        assert type_name_alt == type_name
        assert props_alt == props

        # Dict with string keys and list values
        type_name, props = get_json_schema_type(Dict[str, List[int]])  # noqa: UP006
        assert type_name == 'object'
        assert props['additionalProperties']['type'] == 'array'
        assert props['additionalProperties']['items']['type'] == 'integer'

        type_name_alt, props_alt = get_json_schema_type(dict[str, list[int]])
        assert type_name_alt == type_name
        assert props_alt == props


    # Special type tests
    def test_enum_types(self):
        """Test conversion of Enum types to JSON Schema types."""
        type_name, props = get_json_schema_type(TestEnum)
        assert type_name == 'string'
        assert 'enum' in props
        assert set(props['enum']) == {'red', 'green', 'blue'}


    def test_pydantic_model_types(self):
        """Test conversion of Pydantic model types to JSON Schema types."""
        # Simple model
        type_name, props = get_json_schema_type(Address)
        assert type_name == 'object'
        assert props['type'] == 'object'
        assert 'description' in props

        # Model with nested model
        type_name, props = get_json_schema_type(Person)
        assert type_name == 'object'
        assert props['type'] == 'object'
        assert 'description' in props


    # Edge case tests
    def test_any_type(self):
        """Test conversion of Any type to JSON Schema types."""
        with pytest.raises(ValueError, match="Unsupported type annotation"):
            get_json_schema_type(Any)


    def test_object_type(self):
        """Test conversion of object types to JSON Schema types."""
        # The function should return a reasonable default for unsupported types
        with pytest.raises(ValueError, match="Unsupported type annotation"):
            get_json_schema_type(object)


    def test_tuple_type(self):
        """Test conversion of tuple types to JSON Schema types."""
        # Tuples might be handled as arrays
        type_name, props = get_json_schema_type(tuple[str, int])
        # The exact implementation might vary, but it should return a valid type
        assert type_name in ("array", "string")


    def test_set_type(self):
        """Test conversion of set types to JSON Schema types."""
        # Sets might be handled as arrays
        type_name, props = get_json_schema_type(set[str])
        # The exact implementation might vary, but it should return a valid type
        assert type_name in ("array", "string")


    # Test for unsupported types
    def test_unsupported_types(self):
        """Test handling of types that don't have a direct JSON Schema equivalent."""
        # Custom class
        class CustomClass:
            pass

        with pytest.raises(ValueError, match="Unsupported type annotation"):
            get_json_schema_type(CustomClass)
