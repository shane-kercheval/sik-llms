"""Test the Anthropic Wrapper."""
import asyncio
import os
from pydantic import BaseModel
import pytest
from dotenv import load_dotenv
from sik_llms import (
    Client,
    create_client,
    system_message,
    user_message,
    RegisteredClients,
    Anthropic,
    AnthropicTools,
    TextChunkEvent,
    ResponseSummary,
    ReasoningEffort,
    Tool,
    ToolPredictionResponse,
    ToolPrediction,
    ToolChoice,
    StructuredOutputResponse,
)
from sik_llms.models_base import ThinkingChunkEvent
load_dotenv()

ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'
ANTRHOPIC_THINKING_MODEL = 'claude-3-7-sonnet-latest'


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__async_anthropic_completion_wrapper_call():
    """Test the Anthropic wrapper with multiple concurrent calls."""
    # Create an instance of the wrapper
    client = Anthropic(
        model_name=ANTHROPIC_TEST_MODEL,
        max_tokens=100,
    )
    messages = [
        system_message("You are a helpful assistant."),
        user_message("What is the capital of France?"),
    ]

    async def run_model():  # noqa: ANN202
        chunks = []
        summary = None
        try:
            async for response in client.run_async(messages=messages):
                if isinstance(response, TextChunkEvent):
                    chunks.append(response)
                elif isinstance(response, ResponseSummary):
                    summary = response
            return chunks, summary
        except Exception:
            return [], None

    results = await asyncio.gather(*(run_model() for _ in range(10)))
    passed_tests = []

    for chunks, summary in results:
        response = ''.join([chunk.content for chunk in chunks])
        passed_tests.append(
            'Paris' in response
            and isinstance(summary, ResponseSummary)
            and summary.input_tokens > 0
            and summary.output_tokens > 0
            and summary.input_cost > 0
            and summary.output_cost > 0
            and summary.duration_seconds > 0,
        )

    assert sum(passed_tests) / len(passed_tests) >= 0.9, (
        f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."
    )


def test__Anthropic_registration():
    assert Client.is_registered(RegisteredClients.ANTHROPIC)


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
def test__Anthropic__instantiate():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
    )
    assert isinstance(model, Anthropic)
    assert model.model == ANTHROPIC_TEST_MODEL
    assert model.client is not None
    assert model.client.api_key
    assert model.client.api_key != 'None'


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test__Anthropic__instantiate__run_async():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
    )
    responses = []
    async for response in model.run_async(messages=[user_message("What is the capital of France?")]):  # noqa: E501
        if isinstance(response, TextChunkEvent):
            responses.append(response)

    assert len(responses) > 0
    assert 'Paris' in ''.join([response.content for response in responses])


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
def test__Anthropic_instantiate___parameters():
    model = Client.instantiate(
        client_type=RegisteredClients.ANTHROPIC,
        model_name=ANTHROPIC_TEST_MODEL,
        temperature=0.5,
        max_tokens=100,
    )
    assert isinstance(model, Anthropic)
    assert model.model == ANTHROPIC_TEST_MODEL
    assert model.model_parameters['temperature'] == 0.5
    assert model.model_parameters['max_tokens'] == 100


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
class TestAnthropicReasoning:
    """Test the Anthropic Wrapper with Reasoning."""

    @pytest.mark.parametrize(
        "reasoning_effort",
        [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH],
    )
    def test__Anthropic_instantiate__with_reasoning_effort(self, reasoning_effort: ReasoningEffort):  # noqa: E501
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTHROPIC_TEST_MODEL,
            reasoning_effort=reasoning_effort,
            max_tokens=4000,
        )
        assert isinstance(model, Anthropic)
        assert model.model_parameters.get('thinking') is not None
        assert model.model_parameters['thinking']['type'] == 'enabled'
        assert model.model_parameters['thinking']['budget_tokens']
        assert model.model_parameters['max_tokens'] > model.model_parameters['thinking']['budget_tokens']  # noqa: E501

    def test__Anthropic_instantiate__with_thinking_budget(self):
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTHROPIC_TEST_MODEL,
            thinking_budget_tokens=12000,
            max_tokens=4000,
        )
        assert isinstance(model, Anthropic)
        assert model.model_parameters.get('thinking') is not None
        assert model.model_parameters['thinking']['type'] == 'enabled'
        assert model.model_parameters['thinking']['budget_tokens'] == 12000
        assert model.model_parameters['max_tokens'] > model.model_parameters['thinking']['budget_tokens']  # noqa: E501

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__reasoning_effort(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_THINKING_MODEL,
            reasoning_effort=ReasoningEffort.LOW,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.run_async(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, ResponseSummary):
                # Check that summary contains both thinking and answer
                assert "45" in response.response

        assert has_thinking_content, "No thinking content was generated"
        assert has_text_content, "No text content was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__thinking_budget_tokens(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_THINKING_MODEL,
            thinking_budget_tokens=2000,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False
        has_summary = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.run_async(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, ResponseSummary):
                # Check that summary contains both thinking and answer
                has_summary = True
                assert "45" in response.response

        assert has_thinking_content, "No thinking content was generated"
        assert has_text_content, "No text content was generated"
        assert has_summary, "No summary was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__temperature(self):
        """
        Test that the extended thinking works when temperature is set.

        From docs: "Thinking isn't compatible with temperature, top_p, or top_k modifications as
        well as forced tool use."

        Make sure it doesn't crash.
        """
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_THINKING_MODEL,
            reasoning_effort=ReasoningEffort.LOW,
            temperature=0.5,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.run_async(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, ResponseSummary):
                # Check that summary contains both thinking and answer
                assert "45" in response.response

        assert has_thinking_content, "No thinking content was generated"
        assert has_text_content, "No text content was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__test_redacted_thinking(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_THINKING_MODEL,
            thinking_budget_tokens=2000,
        )
        # Use a prompt that should trigger some thinking content
        has_redacted_thinking_content = False
        has_thinking_content = False
        has_text_content = False
        has_summary = False

        # test redacted thinking:
        # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks
        messages = [user_message("ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB")]  # noqa: E501
        async for response in model.run_async(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                if response.is_redacted:
                    has_redacted_thinking_content = True
                    assert response.content
                else:
                    has_thinking_content = True
            elif isinstance(response, ResponseSummary):
                has_summary = True
                assert response.response

        assert has_redacted_thinking_content, "No redacted thinking content was generated"
        assert not has_thinking_content, "Thinking content was generated"
        assert has_text_content, "No text content was generated"
        assert has_summary, "No summary was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__without_thinking__verify_no_thinking_events(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_THINKING_MODEL,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False
        has_summary = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.run_async(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, ResponseSummary):
                # Check that summary contains both thinking and answer
                has_summary = True

        assert not has_thinking_content, "Thinking content was generated"
        assert has_text_content, "No text content was generated"
        assert has_summary, "No summary was generated"


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
class TestAnthropicTools:
    """Test the OpenAITools Wrapper."""

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__single_tool_single_parameter__instantiate(
            self,
            simple_weather_tool: Tool,
            is_async: bool,
        ):
        """Test calling a simple tool with one required parameter."""
        client = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC_TOOLS,
            model_name=ANTHROPIC_TEST_MODEL,
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

    async def test__no_tool_call(
            self,
            simple_weather_tool: Tool,
        ):
        """Test calling a simple tool when no tool is applicable."""
        client = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC_TOOLS,
            model_name=ANTHROPIC_TEST_MODEL,
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

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__single_tool_multiple_parameters(
            self,
            complex_weather_tool: Tool,
            is_async: bool,
        ):
        """Test calling a tool with multiple parameters including optional ones."""
        client = AnthropicTools(
            model_name=ANTHROPIC_TEST_MODEL,
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

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__multiple_tools(
            self,
            simple_weather_tool: Tool,
            restaurant_tool: Tool,
            is_async: bool,
        ):
        """Test providing multiple tools to the model."""
        client = AnthropicTools(
            model_name=ANTHROPIC_TEST_MODEL,
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

    @pytest.mark.parametrize('is_async', [True, False])
    async def test__enum_parameters(
            self,
            restaurant_tool: Tool,
            is_async: bool,
        ):
        """Test handling of enum parameters."""
        client = AnthropicTools(
            model_name=ANTHROPIC_TEST_MODEL,
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

    async def test__concurrent_tool_calls(self, simple_weather_tool: Tool):
        """Test multiple concurrent tool calls."""
        client = AnthropicTools(
            model_name=ANTHROPIC_TEST_MODEL,
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


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
class TestAnthropicStructuredOutputs:
    """Test the OpenAI Structured Output Wrapper."""

    async def test__anthropic__structured_outputs(self):
        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        client = create_client(
            model_name=ANTHROPIC_TEST_MODEL,
            response_format=CalendarEvent,
        )
        messages=[
            system_message("Extract the event information."),
            user_message("Alice and Bob are going to a science fair on Friday."),
        ]

        response = client(messages=messages)
        assert isinstance(response, ResponseSummary)
        assert isinstance(response.response, StructuredOutputResponse)
        assert isinstance(response.response.parsed, CalendarEvent)
        assert response.response.parsed.name
        assert response.response.parsed.date
        assert response.response.parsed.participants
        assert 'Alice' in response.response.parsed.participants
        assert 'Bob' in response.response.parsed.participants

    async def test__anthropic__structured_outputs__nested(self):
        class Address(BaseModel):
            street: str
            street_2: str | None = None
            city: str
            state: str
            zip_code: str

        class Contact(BaseModel):
            first_name: str
            last_name: str
            phone: str | None = None
            email: str | None = None
            address: Address

        client = create_client(
            model_name=ANTHROPIC_TEST_MODEL,
            response_format=Contact,
        )
        messages=[
            system_message("Extract the information."),
            user_message("Hey my name is Shane Kercheval. I live at 123 Main Street in Anytown, Washington in the USA. The zip code is 12345."),  # noqa: E501
        ]

        response = client(messages=messages)
        assert isinstance(response, ResponseSummary)
        assert isinstance(response.response, StructuredOutputResponse)
        assert isinstance(response.response.parsed, Contact)
        assert response.response.parsed.first_name == 'Shane'
        assert response.response.parsed.last_name == 'Kercheval'
        assert not response.response.parsed.phone
        assert not response.response.parsed.email
        assert response.response.parsed.address.street == '123 Main Street'
        assert not response.response.parsed.address.street_2
        assert response.response.parsed.address.city == 'Anytown'
        assert response.response.parsed.address.state in ('Washington', 'WA')
        assert response.response.parsed.address.zip_code == '12345'
