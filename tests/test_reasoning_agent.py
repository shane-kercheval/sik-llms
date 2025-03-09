"""Tests for the ReasoningAgent class."""
import re
import pytest
from sik_llms.models_base import (
    ErrorEvent,
    Tool,
    Parameter,
    ThinkingEvent,
    ToolPredictionEvent,
    ToolResultEvent,
    ResponseSummary,
)
from sik_llms.reasoning_agent import ReasoningAgent


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
                type='string',
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
                type='string',
                required=True,
                description="The city and state/country (e.g., 'San Francisco, CA')",
            ),
        ],
        func=weather_async,
    )


def test__create_reasoning_prompt(calculator_tool: Tool, test_files_path: str):
    agent = ReasoningAgent(
        model_name='gpt-4o-mini',
        tools=[calculator_tool],
    )
    prompt = agent._create_reasoning_prompt()
    with open(f'{test_files_path}/reasoning_prompt_calculator_tool.txt', 'w') as f:
        f.write(prompt)
    # check that the tool is in the prompt
    assert 'calculator' in prompt
    # check that the parameter is in the prompt
    assert 'expression' in prompt


def test__create_reasoning_prompt__multiple_tools(calculator_tool: Tool, test_files_path: str):
    weather_multi_param_tool = Tool(
        name='get_weather',
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name='location',
                type='string',
                required=True,
                description="The city and state/country (e.g., 'San Francisco, CA')",
            ),
            Parameter(
                name='units',
                type='enum',
                required=True,
                description="Temperature units",
                enum=['°F', '°C'],
            ),
        ],
        func=lambda location, units: f"Weather for {location}: 70{units}, Sunny with some clouds",
    )
    agent = ReasoningAgent(
        model_name='gpt-4o-mini',
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


def test__create_reasoning_prompt__no_tools(test_files_path: str):
    agent = ReasoningAgent(
        model_name='gpt-4o-mini',
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
        model_name='gpt-4o-mini',
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
                type='string',
                required=True,
                description="The city and state/country (e.g., 'San Francisco, CA')",
            ),
        ],
        func=weather_sync,
    )
    agent = ReasoningAgent(
        model_name='gpt-4o-mini',
        tools=[tool],
    )
    # check that the tool is executed successfully
    result = await agent._execute_tool('get_weather', {'location': 'New York, NY'})
    assert result == 'Weather for New York, NY: 72°F, Sunny with some clouds'


@pytest.mark.asyncio
async def test_reasoning_agent_with_calculator(calculator_tool: Tool):
    """Test the ReasoningAgent with a calculator tool using GPT-4o-mini."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[calculator_tool],
        max_iterations=2,
        temperature=0,
    )

    # Test messages
    messages = [{"role": "user", "content": "What is 532 * 124?"}]

    # Run the agent and collect the results
    results = []
    async for result in agent.run_async(messages):
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

    # Last result should be a ResponseSummary with the full response
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"
    assert re.search(r'65[,]?968', last_result.response) is not None, "Final answer should contain 65968 or 65,968"  # noqa: E501

    # Check token accounting
    assert last_result.input_tokens > 0, "Should have input tokens"
    assert last_result.output_tokens > 0, "Should have output tokens"
    assert last_result.duration_seconds > 0, "Should have duration"


@pytest.mark.asyncio
async def test_reasoning_agent_with_multiple_tools(calculator_tool: Tool, weather_tool: Tool):
    """Test the ReasoningAgent with multiple tools."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[calculator_tool, weather_tool],
        temperature=0,
    )

    # Test messages with a complex query requiring multiple tools
    messages = [{
        "role": "user",
        "content": "I'm planning a trip to New York. What's the weather like there? Also, if the temperature in Celsius is 22, what is that in Fahrenheit?",  # noqa: E501
    }]

    # Run the agent and collect the results
    results = []
    async for result in agent.run_async(messages):
        results.append(result)

    # Check that tools were used
    tool_prediction_events = [r for r in results if isinstance(r, ToolPredictionEvent)]

    tool_names = [event.name for event in tool_prediction_events]

    # At least one of the tools should be used
    assert any(name in ["calculator", "get_weather"] for name in tool_names), "Should use at least one tool"  # noqa: E501

    error_events = [r for r in results if isinstance(r, ErrorEvent)]
    assert len(error_events) == 0, "Should not have errors"


    # Last result should be a ResponseSummary with a complete answer
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"

    # The answer should mention weather and temperature conversion
    final_answer = last_result.response
    assert "New York" in final_answer, "Answer should mention New York"
    assert "Fahrenheit" in final_answer, "Answer should mention Fahrenheit"

    # Check token accounting
    assert last_result.input_tokens > 0, "Should have input tokens"
    assert last_result.output_tokens > 0, "Should have output tokens"


@pytest.mark.asyncio
async def test_reasoning_agent_no_tools_needed():
    """Test the ReasoningAgent with a question that doesn't need tools."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[],  # No tools
        max_iterations=2,
        temperature=0,
    )

    # Test messages with a question that doesn't need tools
    messages = [{
        "role": "user",
        "content": "What are three benefits of regular exercise?",
    }]

    # Run the agent and collect the results
    results = []
    async for result in agent.run_async(messages):
        results.append(result)

    # Check thinking occurred
    thinking_events = [r for r in results if isinstance(r, ThinkingEvent)]
    assert len(thinking_events) > 0, "Should have thinking events"

    # Last result should be a ResponseSummary with a complete answer
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"

    # The answer should mention benefits of exercise
    final_answer = last_result.response
    exercise_terms = ["exercise", "health", "fitness", "cardiovascular", "strength", "mental"]
    assert any(term in final_answer.lower() for term in exercise_terms), "Answer should discuss exercise benefits"  # noqa: E501

    # There should be no tool predictions
    tool_prediction_events = [r for r in results if isinstance(r, ToolPredictionEvent)]
    assert len(tool_prediction_events) == 0, "Should not have tool predictions"


@pytest.mark.asyncio
async def test_reasoning_agent_with_lambda_function():
    weather_tool = Tool(
        name='get_weather',
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name='location',
                type='string',
                required=True,
                description="The city (e.g., 'San Francisco')",
            ),
        ],
        func=lambda location: f"Weather for {location}: 72°F, Sunny with some clouds",
    )
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[weather_tool],
        temperature=0,
    )
    messages = [{
        "role": "user",
        "content": "What's the weather like in New York?",
    }]
    results = []
    async for result in agent.run_async(messages):
        results.append(result)
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"
    assert "New York" in last_result.response
    assert "72°F" in last_result.response

    # get to the tool result event
    tool_result_events = [r for r in results if isinstance(r, ToolResultEvent)]
    assert len(tool_result_events) > 0, "Should have tool result events"
    assert tool_result_events[0].result == "Weather for New York: 72°F, Sunny with some clouds"
