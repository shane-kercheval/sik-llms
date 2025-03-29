"""Tests for the Tools Wrappers."""
import asyncio
import os
import pytest
from dotenv import load_dotenv
from enum import Enum
from typing import List, Dict, Literal, Optional, Any, Union  # noqa: UP035
from pydantic import BaseModel, Field
from sik_llms import (
    Client,
    user_message,
    Tool,
    Parameter,
    ToolPredictionResponse,
    ToolPrediction,
    RegisteredClients,
    ToolChoice,
)
from sik_llms.anthropic import _tool_to_anthropic_schema
from sik_llms.models_base import pydantic_model_to_parameters
from sik_llms.openai import _tool_to_openai_schema
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL, ClientConfig

load_dotenv()


class TestPydanticModelToParameters:
    """Tests for pydantic_model_to_parameters function."""

    def test_simple_model(self):
        """Test conversion of a simple Pydantic model."""
        class SimpleModel(BaseModel):
            name: str
            age: int
            active: bool
            a_float: float

        params = pydantic_model_to_parameters(SimpleModel)

        # Check number of parameters
        assert len(params) == 4

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

    def test_primitive_types_to_json_schema(self):
        """Test that Python primitive types are converted to correct JSON Schema types."""
        # Test all primitive types
        class AllTypes(BaseModel):
            a_str: str
            an_int: int
            a_float: float
            a_bool: bool
            a_list: list[str]
            list_floats: list[float]
            a_dict: dict[str, int]

        params = pydantic_model_to_parameters(AllTypes)
        tool = Tool(
            name="test_types",
            parameters=params,
            description="Test type conversions",
            func=lambda **kwargs: str(kwargs),
        )

        # Get the OpenAI format
        openai_format = _tool_to_openai_schema(tool)
        properties = openai_format["function"]["parameters"]["properties"]

        # Check each type conversion
        assert properties["a_str"]["type"] == "string"
        assert properties["an_int"]["type"] == "integer"
        assert properties["a_float"]["type"] == "number"
        assert properties["a_bool"]["type"] == "boolean"
        assert properties["a_list"]["type"] == "array"
        assert properties["a_list"]["items"]["type"] == "string"
        assert properties["list_floats"]["type"] == "array"
        assert properties["list_floats"]["items"]["type"] == "number"
        assert properties["a_dict"]["type"] == "object"
        assert properties["a_dict"]["additionalProperties"]["type"] == "integer"

        # Do the same check for Anthropic format
        anthropic_format = _tool_to_anthropic_schema(tool)
        properties = anthropic_format["input_schema"]["properties"]

        assert properties["a_str"]["type"] == "string"
        assert properties["an_int"]["type"] == "integer"
        assert properties["a_float"]["type"] == "number"
        assert properties["a_bool"]["type"] == "boolean"
        assert properties["a_list"]["type"] == "array"
        assert properties["a_list"]["items"]["type"] == "string"
        assert properties["a_dict"]["type"] == "object"
        assert properties["a_dict"]["additionalProperties"]["type"] == "integer"

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
        assert active_param.required is False, "Fields with defaults should not be marked as required"  # noqa: E501

    def test_nested_model_with_defaults(self):
        """Test conversion of nested models with default values."""
        class Address(BaseModel):
            street: str
            city: str
            country: str = 'USA'  # Default value

        class Person(BaseModel):
            name: str
            phone: str = Field(default='<not provided>')
            address: Address

        params = pydantic_model_to_parameters(Person)

        # Check required status
        name_param = next(p for p in params if p.name == 'name')
        assert name_param.required is True

        phone_param = next(p for p in params if p.name == 'phone')
        assert phone_param.required is False

        address_param = next(p for p in params if p.name == 'address')
        assert address_param.required is True

        # Convert to OpenAI format and check for no defaults
        tool = Tool(
            name='test_person',
            parameters=params,
            description='Test nested defaults',
        )

        openai_format = _tool_to_openai_schema(tool)
        schema = openai_format['function']['parameters']['properties']
        # Verify no default values in schema
        assert 'default' not in schema['phone']
        assert 'street' in schema['address']['properties']
        assert 'city' in schema['address']['properties']
        assert 'country' in schema['address']['properties']
        assert 'default' not in schema['address']['properties']['country']

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
            description="Test app settings",
        )

        openai_format = _tool_to_openai_schema(tool)

        # Verify no defaults exist anywhere in the schema
        def check_no_defaults(obj) -> None:  # noqa: ANN001
            if isinstance(obj, dict):
                assert "default" not in obj, f"Found 'default' in: {obj}"
                for key, value in obj.items():
                    if isinstance(value, dict | list):
                        check_no_defaults(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict | list):
                        check_no_defaults(item)

        check_no_defaults(openai_format)

    def test_model_with_optional_fields(self):
        """Test conversion of a model with Optional fields."""
        class ModelWithOptional(BaseModel):
            name: str
            age: Optional[int]  # Using Optional syntax  # noqa: UP007
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
        assert tags_param.param_type == List[str]  # Direct Python type comparison  # noqa: UP006

        counts_param = next(p for p in params if p.name == 'counts')
        assert counts_param.param_type == Dict[str, int]  # Direct Python type comparison  # noqa: E501, UP006

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
        assert color_param.param_type is str
        assert color_param.valid_values == ['red', 'green', 'blue']

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
        assert person_params[0].valid_values is None

        assert person_params[1].name == 'address'
        assert person_params[1].param_type is Address
        assert person_params[1].required is True
        assert person_params[1].description is not None
        assert "The person's address" in person_params[1].description
        assert person_params[1].valid_values is None

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
        assert items_param.param_type == List[Item]  # noqa: UP006
        assert items_param.description is None

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
        assert tags_param.param_type == list[str]
        assert tags_param.required is False

        metadata_param = next(p for p in params if p.name == 'metadata')
        assert metadata_param.param_type == dict[str, str]
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
        assert Dict[str, int] in data_param.any_of  # noqa: UP006
        assert List[str] in data_param.any_of  # noqa: UP006

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
            data: Union[str, List[int], Dict[str, bool]]  # noqa: UP006, UP007
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
            valid_values=["red", "green", "blue"],
        )

        tool = Tool(
            name="set_color",
            parameters=[param],
            description="Set a color",
            func=lambda color: f"Color set to {color}",
        )

        openai_format = _tool_to_openai_schema(tool)
        assert openai_format["function"]["parameters"]["properties"]["color"]["type"] == "string"
        assert openai_format["function"]["parameters"]["properties"]["color"]["enum"] == ["red", "green", "blue"]  # noqa: E501
        assert openai_format["function"]["parameters"]["properties"]["color"]["description"] == "Color selection"  # noqa: E501

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
            func=lambda id: f"Item {id}",  # noqa: A006
        )

        openai_format = _tool_to_openai_schema(tool)
        assert "anyOf" in openai_format["function"]["parameters"]["properties"]["id"]
        assert len(openai_format["function"]["parameters"]["properties"]["id"]["anyOf"]) == 2
        assert openai_format["function"]["parameters"]["properties"]["id"]["description"] == "ID value"  # noqa: E501
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
            description="Test config",
        )

        # Convert to OpenAI format
        openai_format = _tool_to_openai_schema(tool)
        properties = openai_format["function"]["parameters"]["properties"]

        # Check that no properties have default values
        for prop_name, prop_schema in properties.items():
            assert "default" not in prop_schema, f"Property {prop_name} should not have a default value"  # noqa: E501

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
            valid_values=["red", "green", "blue"],
        )

        tool = Tool(
            name="set_color",
            parameters=[param],
            description="Set a color",
            func=lambda color: f"Color set to {color}",
        )

        anthropic_format = _tool_to_anthropic_schema(tool)
        assert anthropic_format["name"] == "set_color"
        assert anthropic_format["description"] == "Set a color"
        assert anthropic_format["input_schema"]["properties"]["color"]["type"] == "string"
        assert anthropic_format["input_schema"]["properties"]["color"]["enum"] == ["red", "green", "blue"]  # noqa: E501
        assert anthropic_format["input_schema"]["properties"]["color"]["description"] == "Color selection"  # noqa: E501

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
            func=lambda id: f"Item {id}",  # noqa: A006
        )

        anthropic_format = _tool_to_anthropic_schema(tool)
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
            description="Test config",
        )

        # Convert to Anthropic format
        anthropic_format = _tool_to_anthropic_schema(tool)
        properties = anthropic_format["input_schema"]["properties"]

        # Check that no properties have default values
        for prop_name, prop_schema in properties.items():
            assert "default" not in prop_schema, f"Property {prop_name} should not have a default value"  # noqa: E501

        # Check required fields
        required = anthropic_format["input_schema"].get("required", [])
        assert "name" in required
        assert "timeout" not in required
        assert "retries" not in required
        assert "settings" not in required


@pytest.mark.asyncio
@pytest.mark.integration  # these tests make API calls
class TestTools:
    """Test the Tools Wrapper."""

    # @pytest.mark.stochastic(samples=5, threshold=0.5)
    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize('is_async', [
        pytest.param(True, id="Async"),
        pytest.param(False, id="Sync"),
    ])
    async def test__single_tool_single_parameter__instantiate(
            self,
            simple_weather_tool: Tool,
            is_async: bool,
            client_config: ClientConfig,
        ):
        """Test calling a simple tool with one required parameter."""
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[simple_weather_tool],
        )
        if is_async:
            response = await client.run_async(
                messages=[
                    user_message("What's the weather like in Paris?"),
                ],
            )
        else:
            response = client(
                messages=[
                    user_message("What's the weather like in Paris?"),
                ],
            )
        assert isinstance(response, ToolPredictionResponse)
        assert isinstance(response.tool_prediction, ToolPrediction)
        assert response.tool_prediction.name == "get_weather"
        assert "location" in response.tool_prediction.arguments
        assert "Paris" in response.tool_prediction.arguments["location"]
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    async def test__no_tool_call(
            self,
            simple_weather_tool: Tool,
            client_config: ClientConfig,
        ):
        """Test calling a simple tool when no tool is applicable."""
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[simple_weather_tool],
            tool_choice=ToolChoice.AUTO,
        )
        response = await client.run_async(
            messages=[
                user_message("What's the stock price of Apple?"),
            ],
        )
        assert isinstance(response, ToolPredictionResponse)
        assert response.tool_prediction is None
        assert response.message
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize('is_async', [
        pytest.param(True, id="Async"),
        pytest.param(False, id="Sync"),
    ])
    async def test__single_tool_multiple_parameters(
            self,
            complex_weather_tool: Tool,
            is_async: bool,
            client_config: ClientConfig,
        ):
        """Test calling a tool with multiple parameters including optional ones."""
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[complex_weather_tool],
        )
        if is_async:
            response = await client.run_async(
                messages=[
                    user_message("What's the weather like in Tokyo in Celsius with forecast?"),
                ],
            )
        else:
            response = client(
                messages=[
                    user_message("What's the weather like in Tokyo in Celsius with forecast?"),
                ],
            )
        assert response.tool_prediction.name == "get_detailed_weather"
        args = response.tool_prediction.arguments
        assert "Tokyo" in args["location"]
        assert args.get("unit") in ["celsius", "fahrenheit"]
        assert args.get("include_forecast") is not None

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize('is_async', [
        pytest.param(True, id="Async"),
        pytest.param(False, id="Sync"),
    ])
    async def test__multiple_tools(
            self,
            simple_weather_tool: Tool,
            restaurant_tool: Tool,
            is_async: bool,
            client_config: ClientConfig,
        ):
        """Test providing multiple tools to the model."""
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[simple_weather_tool, restaurant_tool],
        )
        # Weather query
        if is_async:
            weather_response = await client.run_async(
                messages=[
                    user_message("What's the weather like in London?"),
                ],
            )
        else:
            weather_response = client(
                messages=[
                    user_message("What's the weather like in London?"),
                ],
            )
        assert weather_response.tool_prediction.name == "get_weather"
        assert "London" in weather_response.tool_prediction.arguments["location"]
        # Restaurant query
        if is_async:
            restaurant_response = await client.run_async(
                messages=[
                    user_message("Find me an expensive Italian restaurant in New York"),
                ],
            )
        else:
            restaurant_response = client(
                messages=[
                    user_message("Find me an expensive Italian restaurant in New York"),
                ],
            )
        assert restaurant_response.tool_prediction.name == "search_restaurants"
        args = restaurant_response.tool_prediction.arguments
        assert "New York" in args["location"]
        assert args.get("cuisine") == "italian"
        assert args.get("price_range") in ["$$", "$$$", "$$$$"]

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize('is_async', [
        pytest.param(True, id="Async"),
        pytest.param(False, id="Sync"),
    ])
    async def test__enum_parameters(
            self,
            restaurant_tool: Tool,
            is_async: bool,
            client_config: ClientConfig,
        ):
        """Test handling of enum parameters."""
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[restaurant_tool],
        )
        test_cases = [
            (
                "Find me a cheap Chinese restaurant in Boston",
                {"cuisine": "chinese", "price_range": "$"},
            ),
            (
                "I want Mexican food in Chicago at a moderate price",
                {"cuisine": "mexican"},
            ),
            (
                "Find an Indian restaurant in Seattle",
                {"cuisine": "indian"},
            ),
        ]

        for prompt, expected_args in test_cases:
            if is_async:
                response = await client.run_async(messages=[user_message(prompt)])
            else:
                response = client(messages=[user_message(prompt)])
            args = response.tool_prediction.arguments
            for key, value in expected_args.items():
                assert args.get(key) == value

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    async def test__concurrent_tool_calls(self, simple_weather_tool: Tool, client_config: ClientConfig):  # noqa: E501
        """Test multiple concurrent tool calls."""
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[simple_weather_tool],
        )

        cities = ["Paris", "London", "Tokyo", "New York", "Sydney"]
        messages = [
            [{"role": "user", "content": f"What's the weather like in {city}?"}]
            for city in cities
        ]

        responses = await asyncio.gather(*(
            client.run_async(messages=msg) for msg in messages
        ))

        for i, response in enumerate(responses):
            assert response.tool_prediction.name == "get_weather"
            assert cities[i] in response.tool_prediction.arguments["location"]
            assert response.input_tokens > 0
            assert response.output_tokens > 0

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize(('prompt', 'expected_status'), [
        pytest.param("Update order #12345 to shipped status", 'shipped', id='shipped'),
        pytest.param("Cancel my order ABC-789", 'cancelled', id='cancelled'),
        pytest.param("Mark order XYZ-456 as delivered", 'delivered', id='delivered'),
        pytest.param("Set order 55555 to processing", 'processing', id='processing'),
    ])
    async def test_tool_with_valid_values(
        self,
        prompt: str,
        expected_status: str,
        client_config: ClientConfig,
    ):
        """Test a tool with parameters that have valid_values constraints."""
        # Define a tool with valid_values (enum-like) constraints
        status_tool = Tool(
            name="update_status",
            description="Update the status of an order",
            parameters=[
                Parameter(
                    name="order_id",
                    param_type=str,
                    required=True,
                    description="The ID of the order to update",
                ),
                Parameter(
                    name="status",
                    param_type=str,
                    required=True,
                    description="The new status for the order",
                    valid_values=["pending", "processing", "shipped", "delivered", "cancelled"],
                ),
            ],
            func=lambda order_id, status: f"Order {order_id} updated to {status}",
        )
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[status_tool],
        )
        response = await client.run_async(
            messages=[user_message(prompt)],
        )
        assert response.tool_prediction.name == "update_status"
        args = response.tool_prediction.arguments
        assert "order_id" in args
        assert "status" in args
        assert args["status"] == expected_status

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize('test_case', [
        pytest.param(
            {
                "prompt": "Look up product with ID 78901",
                "expected_id": 78901,
                "expected_id_type": int,
            },
            id="integer",
        ),
        pytest.param(
            {
                "prompt": "Get details for SKU abc-123",
                "expected_id": "abc-123",
                "expected_id_type": str,
            },
            id="string",
        ),
    ])
    async def test_tool_with_any_of_parameters(
        self,
        test_case: dict,
        client_config: ClientConfig,
        ):
        """Test a tool with parameters that use any_of for union types."""
        # Define a tool with any_of (union type) constraints
        prompt = test_case["prompt"]
        expected_id = test_case["expected_id"]
        expected_id_type = test_case["expected_id_type"]
        search_tool = Tool(
            name="search_item",
            description="Search for an item by ID or name",
            parameters=[
                Parameter(
                    name="identifier",
                    param_type=str,  # Base type
                    required=True,
                    description="The item identifier (can be numeric ID or name string)",
                    any_of=[str, int],  # Can be string or integer
                ),
                Parameter(
                    name="category",
                    param_type=str,
                    required=False,
                    description="Item category to filter by",
                ),
            ],
            func=lambda identifier, category=None: f"Searching for {identifier} in {category or 'all categories'}",  # noqa: E501
        )
        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[search_tool],
        )
        response = await client.run_async(
            messages=[user_message(prompt)],
        )
        assert response.tool_prediction.name == "search_item"
        args = response.tool_prediction.arguments
        assert "identifier" in args

        # Check that the identifier is of the expected type
        assert isinstance(args["identifier"], expected_id_type)
        # Check that it contains the expected value
        if expected_id_type is int:
            assert args["identifier"] == expected_id
        else:
            assert expected_id in args["identifier"]

    @pytest.mark.parametrize('client_config', [
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.OPENAI_TOOLS,
                model_name=OPENAI_TEST_MODEL,
            ),
            id="OpenAI",
        ),
        pytest.param(
            ClientConfig(
                client_type=RegisteredClients.ANTHROPIC_TOOLS,
                model_name=ANTHROPIC_TEST_MODEL,
            ),
            id="Anthropic",
            marks=pytest.mark.skipif(
                os.getenv('ANTHROPIC_API_KEY') is None,
                reason="ANTHROPIC_API_KEY is not set",
            ),
        ),
    ])
    @pytest.mark.parametrize('test_case', [
        pytest.param(
            {
                "prompt": "Calculate shipping cost for products 1001, 1002, and 1003 to 123 Main St, Boston, MA 02108 using express shipping",  # noqa: E501
                "expected_products": [1001, 1002, 1003],
                "expected_city": "Boston",
                "expected_method": "express",
            },
            id="multiple_products_express",
        ),
        pytest.param(
            {
                "prompt": "What's the shipping cost for item 5000 to 456 Park Ave, New York, NY 10022 with standard delivery?",  # noqa: E501
                "expected_products": [5000],
                "expected_city": "New York",
                "expected_method": "standard",
            },
            id="single_product_standard",
        ),
        pytest.param(
            {
                "prompt": "I need overnight shipping for products 7777 and 8888 to 789 Ocean Blvd, Miami, FL 33139",  # noqa: E501
                "expected_products": [7777, 8888],
                "expected_city": "Miami",
                "expected_method": "overnight",
            },
            id="two_products_overnight",
        ),
    ])
    async def test_tool_with_complex_nested_parameters(
        self,
        test_case: dict,
        client_config: ClientConfig,
    ):
        """Test a tool with complex nested parameter structures."""
        # Extract test case values
        prompt = test_case["prompt"]
        expected_products = test_case["expected_products"]
        expected_city = test_case["expected_city"]
        expected_method = test_case["expected_method"]
        # Define a Pydantic model for address
        class Address(BaseModel):
            street: str
            city: str
            zip_code: str
            country: str = "USA"

        # Define a tool that takes a complex parameter
        shipping_tool = Tool(
            name="calculate_shipping",
            description="Calculate shipping cost for an order",
            parameters=[
                Parameter(
                    name="product_ids",
                    param_type=list[int],
                    required=True,
                    description="List of product IDs in the order",
                ),
                Parameter(
                    name="shipping_address",
                    param_type=Address,
                    required=True,
                    description="The delivery address",
                ),
                Parameter(
                    name="shipping_method",
                    param_type=str,
                    required=True,
                    description="Shipping method to use",
                    valid_values=["standard", "express", "overnight"],
                ),
            ],
            func=lambda product_ids, shipping_address, shipping_method: f"Shipping cost for {len(product_ids)} items to {shipping_address.city}: ${len(product_ids) * 5} via {shipping_method}",  # noqa: E501
        )

        client = Client.instantiate(
            client_type=client_config.client_type,
            model_name=client_config.model_name,
            tools=[shipping_tool],
        )
        response = await client.run_async(
            messages=[user_message(prompt)],
        )

        assert response.tool_prediction.name == "calculate_shipping"
        args = response.tool_prediction.arguments

        # Check product_ids
        assert "product_ids" in args
        assert isinstance(args["product_ids"], list)
        assert all(isinstance(_id, int) for _id in args["product_ids"])
        # Check that all expected products are included
        for product_id in expected_products:
            assert product_id in args["product_ids"]

        # Check shipping_address
        assert "shipping_address" in args
        address = args["shipping_address"]
        assert "street" in address
        assert "city" in address
        assert expected_city in address["city"]
        assert "zip_code" in address

        # Check shipping_method
        assert "shipping_method" in args
        assert args["shipping_method"] == expected_method
