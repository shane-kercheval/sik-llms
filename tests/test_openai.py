"""Test the OpenAI Wrapper."""
import asyncio
import pytest
from dotenv import load_dotenv
from sik_llms.models_base import (
    Client,
    system_message,
    user_message,
    Function,
    Parameter,
    FunctionCallResponse,
    FunctionCallResult,
    RegisteredClients,
)
from sik_llms.openai import (
    OpenAI,
    ChatChunkResponse,
    ChatResponseSummary,
    OpenAIFunctions,
)

load_dotenv()

OPENAI_TEST_MODEL = 'gpt-4o-mini'


def test__registration__openai():
    assert Client.is_registered(RegisteredClients.OPENAI)


def test__registration__openai_functions():
    assert Client.is_registered(RegisteredClients.OPENAI_FUNCTIONS)


@pytest.mark.asyncio
class TestOpenAIRegistration:
    """Test the OpenAI Completion Wrapper registration."""

    async def test__openai__instantiate(self):
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name=OPENAI_TEST_MODEL,
        )
        assert isinstance(client, OpenAI)
        assert client.model == OPENAI_TEST_MODEL
        assert client.client is not None
        assert client.client.api_key
        assert client.client.api_key != 'None'
        assert client.client.base_url

        # call async/stream method
        responses = []
        async for response in client.run_async(messages=[user_message("What is the capital of France?")]):  # noqa: E501
            if isinstance(response, ChatChunkResponse):
                responses.append(response)

        assert len(responses) > 0
        assert 'Paris' in ''.join([response.content for response in responses])

        # call non-async method
        response = client(messages=[user_message("What is the capital of France?")])
        assert isinstance(response, ChatResponseSummary)
        assert 'Paris' in response.content

    async def test__openai__compatible_server_instantiate___parameters__missing_server_url(self):
        # This base_url is required for openai-compatible-server
        with pytest.raises(ValueError):  # noqa: PT011
            _ = Client.instantiate(
                client_type=RegisteredClients.OPENAI,
                model_name='openai-compatible-server',
            )

    async def test__openai__compatible_server_instantiate___parameters__missing_max_tokens(self):
        model_config = {
            'server_url': 'http://localhost:8000',
            'temperature': 0.5,
        }
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name='openai-compatible-server',
            **model_config,
        )
        assert isinstance(client, OpenAI)
        assert client.model == 'openai-compatible-server'
        assert client.model_parameters['temperature'] == 0.5
        # we need to set the max_tokens parameter to -1 to avoid it sending just 1 token
        assert client.model_parameters['max_tokens'] is not None

    async def test__openai__compatible_server_instantiate___parameters_max_tokens(self):
        model_config = {
            'server_url': 'http://localhost:8000',
            'max_tokens': 10,
        }
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name='openai-compatible-server',
            **model_config,
        )
        assert isinstance(client, OpenAI)
        assert client.model == 'openai-compatible-server'
        assert client.model_parameters['max_tokens'] == 10

    async def test__openai__functions_instantiate(self):
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI_FUNCTIONS,
            model_name=OPENAI_TEST_MODEL,
            functions=[
                Function(
                    name="get_weather",
                    description="Get the weather for a location.",
                    parameters=[
                        Parameter(
                            name="location",
                            type="string",
                            required=True,
                            description="The city and country for weather info.",
                        ),
                    ],
                ),
            ],
        )
        assert isinstance(client, OpenAIFunctions)
        assert client.model == OPENAI_TEST_MODEL
        assert client.client is not None
        assert client.client.api_key
        assert client.client.api_key != 'None'
        assert client.client.base_url
        assert client


@pytest.mark.asyncio
class TestOpenAI:
    """Test the OpenAI Completion Wrapper."""

    async def test__async_openai_completion_wrapper_call(self):
        # Create an instance of the wrapper
        client = OpenAI(model_name=OPENAI_TEST_MODEL)
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]

        async def run_model():  # noqa: ANN202
            chunks = []
            summary = None
            try:
                async for response in client.run_async(messages=messages):
                    if isinstance(response, ChatChunkResponse):
                        chunks.append(response)
                    elif isinstance(response, ChatResponseSummary):
                        summary = response
                return chunks, summary
            except Exception:
                return [], None

        results = await asyncio.gather(*(run_model() for _ in range(10)))
        passed_tests = []
        for chunks, summary in results:
            assert all(chunk.logprob is not None for chunk in chunks)
            response = ''.join([chunk.content for chunk in chunks])
            passed_tests.append(
                'Paris' in response
                and isinstance(summary, ChatResponseSummary)
                and summary.total_input_tokens > 0
                and summary.total_output_tokens > 0
                and summary.total_input_cost > 0
                and summary.total_output_cost > 0
                and summary.duration_seconds > 0,
            )
        assert sum(passed_tests) / len(passed_tests) >= 0.9, f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."  # noqa: E501


@pytest.mark.asyncio
class TestOpenAIFunctions:
    """Test the OpenAI Function Wrapper."""

    @pytest.fixture
    def simple_weather_function(self):
        """Create a simple weather function with one required parameter."""
        return Function(
            name="get_weather",
            description="Get the weather for a location.",
            parameters=[
                Parameter(
                    name="location",
                    type="string",
                    required=True,
                    description="The city and country for weather info.",
                ),
            ],
        )

    @pytest.fixture
    def complex_weather_function(self):
        """Create a weather function with multiple parameters including an enum."""
        return Function(
            name="get_detailed_weather",
            description="Get detailed weather information for a location.",
            parameters=[
                Parameter(
                    name="location",
                    type="string",
                    required=True,
                    description="The city and country",
                ),
                Parameter(
                    name="unit",
                    type="string",
                    required=False,
                    description="Temperature unit",
                    enum=["celsius", "fahrenheit"],
                ),
                Parameter(
                    name="include_forecast",
                    type="boolean",
                    required=False,
                    description="Whether to include forecast data",
                ),
            ],
        )

    @pytest.fixture
    def restaurant_function(self):
        """Create a restaurant search function with multiple parameters."""
        return Function(
            name="search_restaurants",
            description="Search for restaurants in a location.",
            parameters=[
                Parameter(
                    name="location",
                    type="string",
                    required=True,
                    description="The city to search in",
                ),
                Parameter(
                    name="cuisine",
                    type="string",
                    required=False,
                    description="Type of cuisine",
                    enum=["italian", "chinese", "mexican", "indian"],
                ),
                Parameter(
                    name="price_range",
                    type="string",
                    required=False,
                    description="Price range",
                    enum=["$", "$$", "$$$", "$$$$"],
                ),
                Parameter(
                    name="open_now",
                    type="boolean",
                    required=False,
                    description="Filter for currently open restaurants",
                ),
            ],
        )

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__single_function_single_parameter__instantiate(
            self,
            simple_weather_function: Function,
            is_async: bool,
        ):
        """Test calling a simple function with one required parameter."""
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI_FUNCTIONS,
            model_name=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
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
        assert isinstance(response, FunctionCallResponse)
        assert isinstance(response.function_call, FunctionCallResult)
        assert response.function_call.name == "get_weather"
        assert "location" in response.function_call.arguments
        assert "Paris" in response.function_call.arguments["location"]
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__single_function_multiple_parameters(
            self,
            complex_weather_function: Function,
            is_async: bool,
        ):
        """Test calling a function with multiple parameters including optional ones."""
        client = OpenAIFunctions(
            model_name=OPENAI_TEST_MODEL,
            functions=[complex_weather_function],
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
        assert response.function_call.name == "get_detailed_weather"
        args = response.function_call.arguments
        assert "Tokyo" in args["location"]
        assert args.get("unit") in ["celsius", "fahrenheit"]
        assert args.get("include_forecast") is not None

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__multiple_functions(
            self,
            simple_weather_function: Function,
            restaurant_function: Function,
            is_async: bool,
        ):
        """Test providing multiple functions to the model."""
        client = OpenAIFunctions(
            model_name=OPENAI_TEST_MODEL,
            functions=[simple_weather_function, restaurant_function],
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
        assert weather_response.function_call.name == "get_weather"
        assert "London" in weather_response.function_call.arguments["location"]
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
        assert restaurant_response.function_call.name == "search_restaurants"
        args = restaurant_response.function_call.arguments
        assert "New York" in args["location"]
        assert args.get("cuisine") == "italian"
        assert args.get("price_range") in ["$$", "$$$", "$$$$"]


    @pytest.mark.parametrize('is_async', [True, False])
    async def test__enum_parameters(
            self,
            restaurant_function: Function,
            is_async: bool,
        ):
        """Test handling of enum parameters."""
        client = OpenAIFunctions(
            model_name=OPENAI_TEST_MODEL,
            functions=[restaurant_function],
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
            args = response.function_call.arguments
            for key, value in expected_args.items():
                assert args.get(key) == value

    async def test__concurrent_function_calls(self, simple_weather_function: Function):
        """Test multiple concurrent function calls."""
        client = OpenAIFunctions(
            model_name=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
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
            assert response.function_call.name == "get_weather"
            assert cities[i] in response.function_call.arguments["location"]
            assert response.input_tokens > 0
            assert response.output_tokens > 0

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__function_override(
            self,
            simple_weather_function: Function,
            complex_weather_function: Function,
            is_async: bool,
        ):
        """Test overriding functions during the call."""
        client = OpenAIFunctions(
            model_name=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
        )

        # Override with complex weather function
        if is_async:
            response = await client.run_async(
                messages=[
                    user_message("What's the weather like in Berlin in Fahrenheit?"),
                ],
                functions=[complex_weather_function],
            )
        else:
            response = client(
                messages=[
                    user_message("What's the weather like in Berlin in Fahrenheit?"),
                ],
                functions=[complex_weather_function],
            )

        assert response.function_call.name == "get_detailed_weather"
        args = response.function_call.arguments
        assert "Berlin" in args["location"]
        assert args.get("unit") in ["celsius", "fahrenheit"]
