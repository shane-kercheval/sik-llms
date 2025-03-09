"""Tests for the models_base module."""
import pytest
from enum import Enum
from typing import List, Dict, Literal, Optional, Any, Union  # noqa: UP035
from pydantic import BaseModel, Field

from sik_llms.models_base import Parameter, Tool, pydantic_model_to_parameters, Client, RegisteredClients


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


class TestClientRegistration:
    """Tests for the Client registration system."""

    def test_client_registration(self):
        # Test registration with string
        assert Client.is_registered('MockClient')
        # Test registration with enum
        assert Client.is_registered(RegisteredClients.OPENAI)
        # Test unregistered client
        assert not Client.is_registered('UnknownClient')

    def test_client_instantiation(self):
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

    def test_model_kwargs_not_modified(self):
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
        assert name_param.param_type is str
        assert name_param.required is True

        age_param = next(p for p in params if p.name == 'age')
        assert age_param.param_type is int
        assert age_param.required is True

        active_param = next(p for p in params if p.name == 'active')
        assert active_param.param_type is bool
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
        assert age_param.required is False, "Fields with defaults should not be marked as required"

        active_param = next(p for p in params if p.name == 'active')
        assert active_param.required is False, "Fields with defaults should not be marked as required"

    def test_nested_model_with_defaults(self):
        """Test conversion of nested models with default values."""
        class Address(BaseModel):
            street: str
            city: str
            country: str = "USA"  # Default value
        
        class Person(BaseModel):
            name: str
            address: Address = Field(default_factory=Address)  # Default value using default_factory
            
        params = pydantic_model_to_parameters(Person)
        
        # Check required status
        name_param = next(p for p in params if p.name == 'name')
        assert name_param.required is True
        
        address_param = next(p for p in params if p.name == 'address')
        assert address_param.required is False, "Fields with defaults should not be marked as required"
        
        # Convert to OpenAI format and check for no defaults
        tool = Tool(
            name="test_person",
            parameters=params,
            description="Test nested defaults"
        )
        
        openai_format = tool.to_openai()
        schema = openai_format["function"]["parameters"]["properties"]
        
        # Verify no default values in schema
        assert "default" not in schema["address"]
        # Also verify the address is not in required list
        required_fields = openai_format["function"]["parameters"].get("required", [])
        assert "address" not in required_fields


    def test_complex_nested_objects_with_defaults(self):
        """Test handling of complex nested objects with defaults at multiple levels."""
        class Configuration(BaseModel):
            debug: bool = False
            verbosity: int = 1
        
        class Database(BaseModel):
            url: str
            timeout: int = 30
            config: Configuration = Field(default_factory=Configuration)
        
        class AppSettings(BaseModel):
            app_name: str
            db: Database
            temp_dir: str = "/tmp"
            backup_dbs: list[Database] = Field(default_factory=list)
        
        params = pydantic_model_to_parameters(AppSettings)
        
        # Check required fields
        app_name_param = next(p for p in params if p.name == 'app_name')
        assert app_name_param.required is True
        
        db_param = next(p for p in params if p.name == 'db')
        assert db_param.required is True
        
        temp_dir_param = next(p for p in params if p.name == 'temp_dir')
        assert temp_dir_param.required is False
        
        backup_dbs_param = next(p for p in params if p.name == 'backup_dbs')
        assert backup_dbs_param.required is False
        
        # Convert to OpenAI format
        tool = Tool(
            name="app_settings",
            parameters=params,
            description="Test app settings"
        )
        
        openai_format = tool.to_openai()
        
        # Verify no defaults exist anywhere in the schema
        def check_no_defaults(obj):
            if isinstance(obj, dict):
                assert "default" not in obj, f"Found 'default' in: {obj}"
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        check_no_defaults(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        check_no_defaults(item)
        
        check_no_defaults(openai_format)


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
        assert age_param.param_type is int
        assert age_param.required is True

        email_param = next(p for p in params if p.name == 'email')
        assert email_param.param_type is str
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
        assert tags_param.param_type == List[str]  # Direct Python type comparison

        counts_param = next(p for p in params if p.name == 'counts')
        assert counts_param.param_type == Dict[str, int]  # Direct Python type comparison

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
        assert color_param.param_type is str  # Base type for enum is string
        assert color_param.valid_values == ['red', 'green', 'blue']  # Using valid_values instead of enum

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
        assert person_params[0].param_type is str
        assert person_params[0].required is True
        assert person_params[0].description is None
        assert person_params[0].valid_values is None  # Using valid_values instead of enum

        assert person_params[1].name == 'address'
        assert person_params[1].param_type is Address  # Direct Python type comparison
        assert person_params[1].required is True
        assert person_params[1].description is not None
        assert "The person's address" in person_params[1].description
        assert person_params[1].valid_values is None  # Using valid_values instead of enum

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
        assert items_param.param_type == List[Item]  # Direct Python type comparison
        assert items_param.description is None  # Description likely not included by default

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
        assert tags_param.param_type == list[str]  # Direct Python type comparison
        assert tags_param.required is False

        metadata_param = next(p for p in params if p.name == 'metadata')
        assert metadata_param.param_type == dict[str, str]  # Direct Python type comparison
        assert metadata_param.required is False

    # Add new tests for our updated features
    def test_model_with_union_types(self):
        """Test conversion of a model with union types."""
        class ModelWithUnion(BaseModel):
            id: int | str
            data: Dict[str, int] | List[str]  # noqa: UP006

        params = pydantic_model_to_parameters(ModelWithUnion)

        id_param = next(p for p in params if p.name == 'id')
        assert id_param.param_type is str  # Base type is string (placeholder)
        assert id_param.any_of == [int, str]  # Should contain both union types

        data_param = next(p for p in params if p.name == 'data')
        assert data_param.param_type is str  # Base type is string (placeholder)
        assert len(data_param.any_of) == 2
        assert Dict[str, int] in data_param.any_of
        assert List[str] in data_param.any_of

    def test_model_with_literal_types(self):
        """Test conversion of a model with Literal types."""
        class ModelWithLiteral(BaseModel):
            status: Literal["pending", "approved", "rejected"]
            priority: Literal[1, 2, 3]

        params = pydantic_model_to_parameters(ModelWithLiteral)

        status_param = next(p for p in params if p.name == 'status')
        assert status_param.param_type is str
        assert set(status_param.valid_values) == {"pending", "approved", "rejected"}

        priority_param = next(p for p in params if p.name == 'priority')
        assert priority_param.param_type is int
        assert set(priority_param.valid_values) == {1, 2, 3}


    def test_model_with_mixed_literal_types(self):
        """Test handling of Literal types with mixed value types."""
        class MixedLiteralModel(BaseModel):
            mixed: Literal["success", 404, True]  # String, int, and bool
        
        params = pydantic_model_to_parameters(MixedLiteralModel)
        mixed_param = params[0]
        
        # Since mixed types are converted to strings
        assert mixed_param.param_type is str
        assert set(mixed_param.valid_values) == {"success", "404", "True"}

    def test_model_with_complex_nested_unions(self):
        """Test handling of complex nested union types."""
        class ComplexUnionModel(BaseModel):
            data: Union[str, List[int], Dict[str, bool]]
            data2: str | list[int] | dict[str, bool]
        
        params = pydantic_model_to_parameters(ComplexUnionModel)
        data_param = params[0]
        
        assert data_param.param_type is str  # Base placeholder type
        assert len(data_param.any_of) == 3
        assert str in data_param.any_of
        assert List[int] in data_param.any_of  # noqa: UP006
        assert Dict[str, bool] in data_param.any_of  # noqa: UP006


    def test_parameter_with_param_type_validation(self):
        """Test Parameter class validation for param_type."""
        # Valid param_type values
        Parameter(name="test1", param_type=str, required=True)
        Parameter(name="test2", param_type=int, required=True)
        Parameter(name="test3", param_type=list[str], required=True)
        Parameter(name="test4", param_type=dict[str, int], required=True)
        
        class TestModel(BaseModel):
            x: int
        
        Parameter(name="test5", param_type=TestModel, required=True)
        
        # Invalid param_type values
        with pytest.raises(ValueError):  # noqa: PT011
            Parameter(name="invalid1", param_type=Any, required=True)
        
        with pytest.raises(ValueError):  # noqa: PT011
            Parameter(name="invalid2", param_type="string", required=True)  # `str` instead of type

    def test_parameter_serialization(self):
        """Test Parameter serialization and deserialization."""
        param = Parameter(
            name="test",
            param_type=str, 
            required=True,
            description="Test parameter",
            valid_values=["a", "b", "c"],
        )
        
        # Serialize
        data = param.model_dump()
        
        # Verify serialized data
        assert data["name"] == "test"
        assert "param_type" in data  # Now stored as string representation
        assert data["required"] is True
        assert data["description"] == "Test parameter"
        assert data["valid_values"] == ["a", "b", "c"]
        
        # Deserialize (simplified test)
        deserialized = Parameter.model_validate(data)
        
        # Basic verification of deserialized object
        assert deserialized.name == param.name
        assert deserialized.required == param.required
        assert deserialized.description == param.description
        assert deserialized.valid_values == param.valid_values
        # Note: param_type may not match exactly due to serialization/deserialization

    def test_parameter_to_openai_conversion(self):
        """Test conversion of Parameter objects to OpenAI format."""
        # Test string parameter with valid_values
        param = Parameter(
            name="color",
            param_type=str,
            required=True,
            description="Color selection",
            valid_values=["red", "green", "blue"]
        )
        
        tool = Tool(
            name="set_color",
            parameters=[param],
            description="Set a color",
            func=lambda color: f"Color set to {color}"
        )
        
        openai_format = tool.to_openai()
        assert openai_format["function"]["parameters"]["properties"]["color"]["type"] == "string"
        assert openai_format["function"]["parameters"]["properties"]["color"]["enum"] == ["red", "green", "blue"]
        assert openai_format["function"]["parameters"]["properties"]["color"]["description"] == "Color selection"

    def test_parameter_to_openai_with_any_of(self):
        """Test conversion of Parameter with any_of to OpenAI format."""
        param = Parameter(
            name="id",
            param_type=str,  # Base type
            required=True,
            description="ID value",
            any_of=[str, int],
        )
        
        tool = Tool(
            name="get_item",
            parameters=[param],
            description="Get an item",
            func=lambda id: f"Item {id}",
        )

        openai_format = tool.to_openai()
        assert "anyOf" in openai_format["function"]["parameters"]["properties"]["id"]
        assert len(openai_format["function"]["parameters"]["properties"]["id"]["anyOf"]) == 2
        assert openai_format["function"]["parameters"]["properties"]["id"]["description"] == "ID value"
        assert openai_format['function']["description"] == "Get an item"

    def test_openai_schema_has_no_defaults(self):
        """Test that the OpenAI schema doesn't contain any default values."""
        class ConfigModel(BaseModel):
            name: str
            timeout: int = 30
            retries: int = 3
            settings: dict = Field(default_factory=dict)
        
        params = pydantic_model_to_parameters(ConfigModel)
        tool = Tool(
            name="test_config",
            parameters=params,
            description="Test config"
        )
        
        # Convert to OpenAI format
        openai_format = tool.to_openai()
        properties = openai_format["function"]["parameters"]["properties"]
        
        # Check that no properties have default values
        for prop_name, prop_schema in properties.items():
            assert "default" not in prop_schema, f"Property {prop_name} should not have a default value"
        
        # Check required fields
        required = openai_format["function"]["parameters"].get("required", [])
        assert "name" in required
        assert "timeout" not in required
        assert "retries" not in required
        assert "settings" not in required


    def test_parameter_to_anthropic_conversion(self):
        """Test conversion of Parameter objects to Anthropic format."""
        param = Parameter(
            name="color",
            param_type=str,
            required=True,
            description="Color selection",
            valid_values=["red", "green", "blue"]
        )
        
        tool = Tool(
            name="set_color",
            parameters=[param],
            description="Set a color",
            func=lambda color: f"Color set to {color}"
        )
        
        anthropic_format = tool.to_anthropic()
        assert anthropic_format["name"] == "set_color"
        assert anthropic_format["description"] == "Set a color"
        assert anthropic_format["input_schema"]["properties"]["color"]["type"] == "string"
        assert anthropic_format["input_schema"]["properties"]["color"]["enum"] == ["red", "green", "blue"]
        assert anthropic_format["input_schema"]["properties"]["color"]["description"] == "Color selection"

    def test_parameter_to_anthropic_with_any_of(self):
        """Test conversion of Parameter with any_of to Anthropic format."""
        param = Parameter(
            name="id",
            param_type=str,  # Base type
            required=True,
            description="ID value",
            any_of=[str, int],
        )
        
        tool = Tool(
            name="get_item",
            parameters=[param],
            description="Get an item",
            func=lambda id: f"Item {id}",
        )

        anthropic_format = tool.to_anthropic()
        assert "anyOf" in anthropic_format["input_schema"]["properties"]["id"]
        assert len(anthropic_format["input_schema"]["properties"]["id"]["anyOf"]) == 2
        assert anthropic_format["input_schema"]["properties"]["id"]["description"] == "ID value"

    def test_anthropic_schema_has_no_defaults(self):
        """Test that the Anthropic schema doesn't contain any default values."""
        class ConfigModel(BaseModel):
            name: str
            timeout: int = 30
            retries: int = 3
            settings: dict = Field(default_factory=dict)
        
        params = pydantic_model_to_parameters(ConfigModel)
        tool = Tool(
            name="test_config",
            parameters=params,
            description="Test config"
        )
        
        # Convert to Anthropic format
        anthropic_format = tool.to_anthropic()
        properties = anthropic_format["input_schema"]["properties"]
        
        # Check that no properties have default values
        for prop_name, prop_schema in properties.items():
            assert "default" not in prop_schema, f"Property {prop_name} should not have a default value"
        
        # Check required fields
        required = anthropic_format["input_schema"].get("required", [])
        assert "name" in required
        assert "timeout" not in required
        assert "retries" not in required
        assert "settings" not in required
