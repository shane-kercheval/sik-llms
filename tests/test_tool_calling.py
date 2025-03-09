"""Test the OpenAI Wrapper."""
import asyncio
import os
from pydantic import BaseModel
import pytest
from dotenv import load_dotenv
from sik_llms import (
    Client,
    user_message,
    Tool,
    Parameter,
    ToolPredictionResponse,
    ToolPrediction,
    RegisteredClients,
)
from sik_llms.models_base import ToolChoice
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL, ClientConfig

load_dotenv()


@pytest.mark.asyncio
class TestTools:
    """Test the Tools Wrapper."""

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

    @pytest.mark.asyncio
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
