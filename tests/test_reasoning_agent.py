"""Tests for the ReasoningAgent class."""
import os
import re
import pytest
from sik_llms import (
    user_message,
    ErrorEvent,
    Tool,
    Parameter,
    ThinkingEvent,
    ToolPredictionEvent,
    ToolResultEvent,
    TextResponse,
    ReasoningAgent,
)
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL
from dotenv import load_dotenv
load_dotenv()


async def calculator_async(expression: str) -> str:
        """Execute calculator tool."""
        try:
            # Only allow simple arithmetic for safety
            allowed_chars = set('0123456789+-*/() .')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e!s}"


async def weather_async(location: str) -> str:
    """Mock weather tool - returns fake data."""
    # Return mock weather data
    return f"Weather for {location}: 72°F, Sunny with some clouds"


@pytest.fixture
def calculator_tool():
    """Fixture for a calculator tool."""
    return Tool(
        name='calculator',
        description="Perform mathematical calculations",
        parameters=[
            Parameter(
                name='expression',
                param_type=str,
                required=True,
                description="The mathematical expression to evaluate (e.g., '2 + 2', '5 * 10')",
            ),
        ],
        func=calculator_async,
    )


@pytest.fixture
def weather_tool():
    """Fixture for a weather tool."""
    return Tool(
        name='get_weather',
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name='location',
                param_type=str,
                required=True,
                description="The city and state/country (e.g., 'San Francisco, CA')",
            ),
        ],
        func=weather_async,
    )


def test__create_reasoning_prompt(calculator_tool: Tool, test_files_path: str):
    agent = ReasoningAgent(
        model_name=OPENAI_TEST_MODEL,
        tools=[calculator_tool],
    )
    prompt = agent._create_reasoning_prompt()
    with open(f'{test_files_path}/reasoning_prompt_calculator_tool.txt', 'w') as f:
        f.write(prompt)
    # check that the tool is in the prompt
    assert 'calculator' in prompt
    # check that the parameter is in the prompt
    assert 'expression' in prompt


def test__create_reasoning_prompt__multiple_tools(
        calculator_tool: Tool,
        test_files_path: str,
    ):
    weather_multi_param_tool = Tool(
        name='get_weather',
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name='location',
                param_type=str,
                required=True,
                description="The city and state/country (e.g., 'San Francisco, CA')",
            ),
            Parameter(
                name='units',
                param_type=str,
                required=True,
                description="Temperature units",
                enum=['°F', '°C'],
            ),
        ],
        func=lambda location, units: f"Weather for {location}: 70{units}, Sunny with some clouds",
    )
    agent = ReasoningAgent(
        model_name=OPENAI_TEST_MODEL,
        tools=[calculator_tool, weather_multi_param_tool],
    )
    prompt = agent._create_reasoning_prompt()
    with open(f'{test_files_path}/reasoning_prompt_calculator_weather_tool.txt', 'w') as f:
        f.write(prompt)
    # check that the calculator tool is in the prompt
    assert 'calculator' in prompt
    # check that the parameter is in the prompt
    assert 'expression' in prompt
    # check that the weather tool is in the prompt
    assert 'get_weather' in prompt
    # check that the parameter is in the prompt
    assert 'location' in prompt


@pytest.mark.parametrize('model_name', [
    pytest.param(
        OPENAI_TEST_MODEL,
        id="OpenAI",
    ),
    pytest.param(
        ANTHROPIC_TEST_MODEL,
        id="Anthropic",
        marks=pytest.mark.skipif(
            os.getenv('ANTHROPIC_API_KEY') is None,
            reason="ANTHROPIC_API_KEY is not set",
        ),
    ),
])
def test__create_reasoning_prompt__no_tools(test_files_path: str, model_name: str):
    agent = ReasoningAgent(
        model_name=model_name,
        tools=[],
    )
    # check that the prompt signals that there are no tools
    prompt = agent._create_reasoning_prompt()
    with open(f'{test_files_path}/reasoning_prompt_no_tools.txt', 'w') as f:
        f.write(prompt)
    assert 'No tools' in prompt


@pytest.mark.asyncio
async def test__execute_tool__async(calculator_tool: Tool):
    agent = ReasoningAgent(
        model_name=OPENAI_TEST_MODEL,
        tools=[calculator_tool],
    )
    # check that the tool is executed successfully
    result = await agent._execute_tool('calculator', {'expression': '234 * 22'})
    assert result == '5148'


@pytest.mark.asyncio
async def test__execute_tool__sync():
    def weather_sync(location: str) -> str:
        """Mock weather tool - returns fake data."""
        # Return mock weather data
        return f"Weather for {location}: 72°F, Sunny with some clouds"

    tool = Tool(
        name='get_weather',
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name='location',
                param_type=str,
                required=True,
                description="The city and state/country (e.g., 'San Francisco, CA')",
            ),
        ],
        func=weather_sync,
    )
    agent = ReasoningAgent(
        model_name=OPENAI_TEST_MODEL,
        tools=[tool],
    )
    # check that the tool is executed successfully
    result = await agent._execute_tool('get_weather', {'location': 'New York, NY'})
    assert result == 'Weather for New York, NY: 72°F, Sunny with some clouds'


@pytest.mark.asyncio
@pytest.mark.integration
# @pytest.mark.stochastic(samples=5, threshold=0.5)
@pytest.mark.parametrize('model_name', [
    pytest.param(
        OPENAI_TEST_MODEL,
        id="OpenAI",
    ),
    pytest.param(
        ANTHROPIC_TEST_MODEL,
        id="Anthropic",
        marks=pytest.mark.skipif(
            os.getenv('ANTHROPIC_API_KEY') is None,
            reason="ANTHROPIC_API_KEY is not set",
        ),
    ),
])
async def test_reasoning_agent_with_calculator(calculator_tool: Tool, model_name: str):
    """Test the ReasoningAgent with a calculator tool using GPT-4o-mini."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name=model_name,
        tools=[calculator_tool],
        max_iterations=2,
        temperature=0,
    )

    results = []
    async for result in agent.stream([user_message("What is 532 * 124?")]):
        results.append(result)

    # Check that we got the expected results
    thinking_events = [r for r in results if isinstance(r, ThinkingEvent)]
    assert len(thinking_events) > 0, "Should have thinking events"

    error_events = [r for r in results if isinstance(r, ErrorEvent)]
    assert len(error_events) == 0, "Should not have errors"

    tool_prediction_events = [r for r in results if isinstance(r, ToolPredictionEvent)]
    assert len(tool_prediction_events) > 0, "Should have tool prediction events"
    assert tool_prediction_events[0].name == "calculator", "Should use calculator tool"

    tool_result_events = [r for r in results if isinstance(r, ToolResultEvent)]
    assert len(tool_result_events) > 0, "Should have tool result events"
    assert "65968" in tool_result_events[0].result, "Result should be 65968"

    # Last result should be a TextResponse with the full response
    last_result = results[-1]
    assert isinstance(last_result, TextResponse), "Last result should be TextResponse"
    assert re.search(r'65[,]?968', last_result.response) is not None, "Final answer should contain 65968 or 65,968"  # noqa: E501

    # Check token accounting
    assert last_result.input_tokens > 0, "Should have input tokens"
    assert last_result.output_tokens > 0, "Should have output tokens"
    assert last_result.duration_seconds > 0, "Should have duration"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reasoning_agent__no_final_answer(calculator_tool: Tool):
    """Test the ReasoningAgent with a calculator tool using GPT-4o-mini."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name=OPENAI_TEST_MODEL,
        tools=[calculator_tool],
        max_iterations=2,
        generate_final_response=False,
        temperature=0,
    )

    results = []
    async for result in agent.stream([user_message("What is 532 * 124?")]):
        results.append(result)

    # Check that we got the expected results
    thinking_events = [r for r in results if isinstance(r, ThinkingEvent)]
    assert len(thinking_events) > 0, "Should have thinking events"

    error_events = [r for r in results if isinstance(r, ErrorEvent)]
    assert len(error_events) == 0, "Should not have errors"

    tool_prediction_events = [r for r in results if isinstance(r, ToolPredictionEvent)]
    assert len(tool_prediction_events) > 0, "Should have tool prediction events"
    assert tool_prediction_events[0].name == "calculator", "Should use calculator tool"

    tool_result_events = [r for r in results if isinstance(r, ToolResultEvent)]
    assert len(tool_result_events) > 0, "Should have tool result events"
    assert "65968" in tool_result_events[0].result, "Result should be 65968"

    # Last result should be a TextResponse with the full response
    last_result = results[-1]
    assert isinstance(last_result, TextResponse), "Last result should be TextResponse"
    # generate_final_response=False, so there should be no final answer that was generated
    assert not last_result.response

    # Check token accounting
    assert last_result.input_tokens > 0, "Should have input tokens"
    assert last_result.output_tokens > 0, "Should have output tokens"
    assert last_result.duration_seconds > 0, "Should have duration"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.stochastic(samples=5, threshold=0.5)
@pytest.mark.parametrize('model_name', [
    pytest.param(
        OPENAI_TEST_MODEL,
        id="OpenAI",
    ),
    pytest.param(
        ANTHROPIC_TEST_MODEL,
        id="Anthropic",
        marks=pytest.mark.skipif(
            os.getenv('ANTHROPIC_API_KEY') is None,
            reason="ANTHROPIC_API_KEY is not set",
        ),
    ),
])
@pytest.mark.parametrize('location', [
    pytest.param(
        "New York",
        id="New York",
    ),
    pytest.param(
        "London",
        id="London - Returns None",
    ),
])
async def test_reasoning_agent__with_non_string_tool_return_values(model_name: str, location: str):

    # this function returns a dict if the location is found, otherwise None
    async def weather(location: str, units: str) -> str:
        weather_data = {
            'New York': '68',
            'San Francisco': '62',
            'Miami': '85',
            'Chicago': '55',
            'Los Angeles': '75',
        }
        for city in weather_data:  # noqa: PLC0206
            if city.lower() in location.lower():
                temp = weather_data[city]
                if units == 'C':
                    # C = (°F - 32) x (5/9)

                    temp = round((temp - 32) * 5 / 9)
                return {location: f"{temp}°{units}"}
        return None

    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name="location",
                param_type=str,
                required=True,
                description="The name of the city (e.g., 'San Francisco', 'New York', 'London')",
            ),
            Parameter(
                name='units',
                param_type=str,
                required=True,
                description="The units for temperature",
                enum=['F', 'C'],
            ),
        ],
        func=weather,
    )

    agent = ReasoningAgent(
        model_name=model_name,
        tools=[weather_tool],
        max_iterations=1,
    )
    messages = [user_message(f"What's the weather like in {location}?")]
    results = []
    async for result in agent.stream(messages):
        results.append(result)
    last_result = results[-1]

    tool_result_events = [r for r in results if isinstance(r, ToolResultEvent)]
    if len(tool_result_events) != 1:
        print(f"tool_result_events: `{tool_result_events}`")
    assert len(tool_result_events) == 1
    if location == "London":
        # London is not in the weather data, so the tool should return None
        # and the reasoning agent should not crash
        assert tool_result_events[0].result is None
    else:
        assert location in tool_result_events[0].result

    assert isinstance(last_result, TextResponse)
    if location == "London":
        assert last_result.response
    else:
        assert location in last_result.response


@pytest.mark.asyncio
@pytest.mark.integration
# @pytest.mark.stochastic(samples=5, threshold=0.5)
@pytest.mark.parametrize('tools', [[], None])
async def test_reasoning_agent_no_tools_needed(tools: list | None):
    """Test the ReasoningAgent with a question that doesn't need tools."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name=OPENAI_TEST_MODEL,
        tools=tools,  # No tools
        max_iterations=2,
        temperature=0,
    )
    results = []
    async for result in agent.stream([user_message("Write a very short haiku.")]):
        results.append(result)

    # Check thinking occurred
    thinking_events = [r for r in results if isinstance(r, ThinkingEvent)]
    assert len(thinking_events) > 0, "Should have thinking events"

    # Last result should be a TextResponse with a complete answer
    last_result = results[-1]
    assert isinstance(last_result, TextResponse), "Last result should be TextResponse"

    # There should be no tool predictions
    tool_prediction_events = [r for r in results if isinstance(r, ToolPredictionEvent)]
    assert len(tool_prediction_events) == 0, "Should not have tool predictions"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.stochastic(samples=5, threshold=0.5)
@pytest.mark.parametrize('model_name', [
    pytest.param(
        OPENAI_TEST_MODEL,
        id="OpenAI",
    ),
    pytest.param(
        ANTHROPIC_TEST_MODEL,
        id="Anthropic",
        marks=pytest.mark.skipif(
            os.getenv('ANTHROPIC_API_KEY') is None,
            reason="ANTHROPIC_API_KEY is not set",
        ),
    ),
])
async def test_reasoning_agent_with_lambda_function(model_name: str):
    weather_tool = Tool(
        name='get_weather',
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name='location',
                param_type=str,
                required=True,
                description="The city (e.g., 'San Francisco')",
            ),
        ],
        func=lambda location: f"Weather for {location}: 72°F, Sunny with some clouds",
    )
    agent = ReasoningAgent(
        model_name=model_name,
        tools=[weather_tool],
        temperature=0,
    )
    results = []
    async for result in agent.stream([user_message("What's the weather like in New York?")]):
        results.append(result)
    last_result = results[-1]
    assert isinstance(last_result, TextResponse), "Last result should be TextResponse"
    assert "New York" in last_result.response
    assert "72°F" in last_result.response

    # get to the tool result event
    tool_result_events = [r for r in results if isinstance(r, ToolResultEvent)]
    assert len(tool_result_events) > 0, "Should have tool result events"
    assert tool_result_events[0].result == "Weather for New York: 72°F, Sunny with some clouds"
