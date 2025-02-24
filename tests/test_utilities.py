"""Tests for the utilities module."""
import pytest
from enum import Enum, auto
from sik_llms.utilities import Registry


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

