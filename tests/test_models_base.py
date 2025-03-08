"""Tests for the models_base module."""
import pytest
from enum import Enum
from typing import List, Dict, Optional, Any  # noqa: UP035
from pydantic import BaseModel, Field

from sik_llms.models_base import pydantic_model_to_parameters, Client, RegisteredClients


class TestPydanticModelToParameters:
    """Tests for pydantic_model_to_parameters function."""

    def test_simple_model(self):
        """Test conversion of a simple Pydantic model."""
        class SimpleModel(BaseModel):
            name: str
            age: int
            active: bool

        params = pydantic_model_to_parameters(SimpleModel)

        # Check number of parameters
        assert len(params) == 3

        # Check parameter properties
        name_param = next(p for p in params if p.name == 'name')
        assert name_param.type == 'string'
        assert name_param.required is True

        age_param = next(p for p in params if p.name == 'age')
        assert age_param.type == 'integer'
        assert age_param.required is True

        active_param = next(p for p in params if p.name == 'active')
        assert active_param.type == 'boolean'
        assert active_param.required is True

    def test_model_with_defaults(self):
        """Test conversion of a model with default values."""
        class ModelWithDefaults(BaseModel):
            name: str
            age: int = 30
            active: bool = True

        params = pydantic_model_to_parameters(ModelWithDefaults)

        # Check required status
        name_param = next(p for p in params if p.name == 'name')
        assert name_param.required is True

        age_param = next(p for p in params if p.name == 'age')
        assert age_param.required is False

        active_param = next(p for p in params if p.name == 'active')
        assert active_param.required is False

    def test_model_with_optional_fields(self):
        """Test conversion of a model with Optional fields."""
        class ModelWithOptional(BaseModel):
            name: str
            age: Optional[int]  # Using Optional syntax
            email: str | None   # Using pipe syntax

        ModelWithOptional(name='shane', email=None, age=None)

        params = pydantic_model_to_parameters(ModelWithOptional)

        # Check required status
        name_param = next(p for p in params if p.name == 'name')
        assert name_param.required is True

        age_param = next(p for p in params if p.name == 'age')
        assert age_param.type == 'integer'
        assert age_param.required is True

        email_param = next(p for p in params if p.name == 'email')
        assert email_param.type == 'string'
        assert email_param.required is True

    def test_model_with_descriptions(self):
        """Test conversion of a model with field descriptions."""
        class ModelWithDescriptions(BaseModel):
            name: str = Field(description="The person's full name")
            age: int = Field(description='Age in years')

        params = pydantic_model_to_parameters(ModelWithDescriptions)

        name_param = next(p for p in params if p.name == 'name')
        assert name_param.description == "The person's full name"

        age_param = next(p for p in params if p.name == 'age')
        assert age_param.description == 'Age in years'

    def test_model_with_complex_types(self):
        """Test conversion of a model with complex field types."""
        class ModelWithComplexTypes(BaseModel):
            tags: List[str]  # noqa: UP006
            counts: Dict[str, int]  # noqa: UP006

        params = pydantic_model_to_parameters(ModelWithComplexTypes)

        tags_param = next(p for p in params if p.name == 'tags')
        assert tags_param.type == 'array'

        counts_param = next(p for p in params if p.name == 'counts')
        assert counts_param.type == 'object'

    def test_model_with_enum(self):
        """Test conversion of a model with Enum fields."""
        class Color(Enum):
            RED = 'red'
            GREEN = 'green'
            BLUE = 'blue'

        class ModelWithEnum(BaseModel):
            color: Color

        params = pydantic_model_to_parameters(ModelWithEnum)

        color_param = params[0]
        assert color_param.type == 'enum'
        assert set(color_param.enum) == {'red', 'green', 'blue'}

    # Better test for nested models
    def test_nested_models(self):
        """Test conversion of nested Pydantic models."""
        class Address(BaseModel):
            street: str
            city: str
            zip_code: str

        class Person(BaseModel):
            name: str
            address: Address = Field(description="The person's address")

        person_params = pydantic_model_to_parameters(Person)
        assert person_params

        assert person_params[0].name == 'name'
        assert person_params[0].type == 'string'
        assert person_params[0].required is True
        assert person_params[0].description is None
        assert person_params[0].enum is None

        assert person_params[1].name == 'address'
        assert person_params[1].type == 'object'
        assert person_params[1].required is True
        assert person_params[1].description is not None
        assert "The person's address" in person_params[1].description
        assert person_params[1].enum is None
        assert 'Address' in person_params[1].description
        assert 'street' in person_params[1].description
        assert 'city' in person_params[1].description
        assert 'zip_code' in person_params[1].description


    # Better test for list of models
    def test_model_with_list_of_models(self):
        """Test conversion of a model with a list of nested models."""
        class Item(BaseModel):
            name: str
            price: float

        class Order(BaseModel):
            order_id: str
            items: List[Item]  # noqa: UP006

        params = pydantic_model_to_parameters(Order)

        items_param = next(p for p in params if p.name == 'items')
        assert items_param.type == 'array'
        assert items_param.description is not None
        assert 'Item' in items_param.description  # Check it contains the model name
        assert 'array' in items_param.description.lower()  # Check it mentions it's an array

    def test_model_with_unsupported_types(self):
        """Test handling of models with unsupported types."""
        class ModelWithUnsupportedTypes(BaseModel):
            name: str
            anything: Any  # Unsupported type

        with pytest.raises(ValueError):  # noqa: PT011
            pydantic_model_to_parameters(ModelWithUnsupportedTypes)

    def test_model_with_optional_complex_types(self):
        """Test conversion of a model with optional complex types."""
        class ModelWithOptionalComplex(BaseModel):
            tags: list[str] | None = None
            metadata: dict[str, str] | None = None

        params = pydantic_model_to_parameters(ModelWithOptionalComplex)

        tags_param = next(p for p in params if p.name == 'tags')
        assert tags_param.type == 'array'
        assert tags_param.required is False

        metadata_param = next(p for p in params if p.name == 'metadata')
        assert metadata_param.type == 'object'
        assert metadata_param.required is False


class MockClient(Client):  # noqa: D101
    def __init__(self, model_name: str, **kwargs):  # noqa: ANN003
        self.model_name = model_name
        self.kwargs = kwargs

    async def run_async(self, messages):  # noqa: ANN001
        # Simple implementation for testing
        return messages


@Client.register('MockClient')
class RegisteredMockClient(MockClient):  # noqa: D101
    pass


def test_client_registration():
    # Test registration with string
    assert Client.is_registered('MockClient')
    # Test registration with enum
    assert Client.is_registered(RegisteredClients.OPENAI)
    # Test unregistered client
    assert not Client.is_registered('UnknownClient')


def test_client_instantiation():
    # Test instantiation with string registration
    client = Client.instantiate(
        'MockClient',
        model_name='test-model',
        temperature=0.7,
    )
    assert isinstance(client, RegisteredMockClient)
    assert client.model_name == 'test-model'
    assert client.kwargs['temperature'] == 0.7

    # Test instantiation with enum registration
    client = Client.instantiate(
        RegisteredClients.OPENAI,
        model_name='gpt-4o',
        max_tokens=100,
    )
    assert client.model == 'gpt-4o'
    assert client.model_parameters['max_tokens'] == 100


def test_model_kwargs_not_modified():
    original_kwargs = {'temperature': 0.7, 'max_tokens': 100}
    kwargs_copy = original_kwargs.copy()

    client = Client.instantiate(
        'MockClient',
        model_name='test-model',
        **original_kwargs,
    )

    # Verify original kwargs weren't modified
    assert original_kwargs == kwargs_copy

    # Verify client has correct kwargs
    assert client.kwargs['temperature'] == 0.7
    assert client.kwargs['max_tokens'] == 100
    assert client.model_name == 'test-model'

    # Modify the original kwargs and verify client kwargs are unchanged
    original_kwargs['temperature'] = 0.5
    original_kwargs['max_tokens'] = 200
    assert client.kwargs['temperature'] == 0.7
    assert client.kwargs['max_tokens'] == 100
