import re
import pytest
import asyncio
from typing import Dict, Any, List

from sik_llms.models_base import (
    Tool, Parameter, ResponseChunk, ResponseSummary, ContentType
)
from sik_llms.reasoning_agent import ReasoningAgent


@pytest.fixture
def calculator_tool():
    """Fixture for a calculator tool."""
    return Tool(
        name="calculator",
        description="Perform mathematical calculations",
        parameters=[
            Parameter(
                name="expression", 
                type="string", 
                required=True, 
                description="The mathematical expression to evaluate (e.g., '2 + 2', '5 * 10')"
            )
        ]
    )


@pytest.fixture
def weather_tool():
    """Fixture for a weather tool."""
    return Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters=[
            Parameter(
                name="location", 
                type="string", 
                required=True, 
                description="The city and state/country (e.g., 'San Francisco, CA')"
            )
        ]
    )


@pytest.fixture
def tool_executors():
    """Fixture for tool executors."""
    
    async def calculator_executor(expression: str) -> str:
        """Execute calculator tool."""
        try:
            # Only allow simple arithmetic for safety
            allowed_chars = set("0123456789+-*/() .")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def weather_executor(location: str) -> str:
        """Mock weather tool - returns fake data."""
        # Return mock weather data
        return f"Weather for {location}: 72°F, Sunny with some clouds"
    
    return {
        "calculator": calculator_executor,
        "get_weather": weather_executor
    }


@pytest.mark.asyncio
async def test_reasoning_agent_with_calculator(calculator_tool, tool_executors):
    """Test the ReasoningAgent with a calculator tool using GPT-4o-mini."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[calculator_tool],
        tool_executors=tool_executors,
        max_iterations=2,
        temperature=0
    )
    
    # Test messages
    messages = [{"role": "user", "content": "What is 532 * 124?"}]
    
    # Run the agent and collect the results
    results = []
    async for result in agent.run_async(messages):
        results.append(result)
    
    # Check that we got the expected results
    thinking_chunks = [
        r for r in results 
        if isinstance(r, ResponseChunk) and r.content_type == ContentType.THINKING
    ]
    assert len(thinking_chunks) > 0, "Should have thinking chunks"
    
    tool_predictions = [
        r for r in results 
        if isinstance(r, ResponseChunk) and r.content_type == ContentType.TOOL_PREDICTION
    ]
    assert len(tool_predictions) > 0, "Should have tool prediction chunks"
    assert "calculator" in tool_predictions[0].content, "Should use calculator tool"
    
    tool_results = [
        r for r in results 
        if isinstance(r, ResponseChunk) and r.content_type == ContentType.TOOL_RESULT
    ]
    assert len(tool_results) > 0, "Should have tool result chunks"
    assert "65968" in tool_results[0].content, "Result should be 65968"

    iteration = 0
    for r in results[0:-1]:
        if r.iteration != iteration:
            iteration = r.iteration
            print(f"\n\n[ITERATION {iteration}]:\n\n")
        print(r.content, end='')
    
    # Last result should be a ResponseSummary with the full response
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"
    assert re.search(r'65[,]?968', last_result.content) is not None, "Result should be 65968 or 65,968"
    

    ResponseSummary

    # Check token accounting
    assert last_result.input_tokens > 0, "Should have input tokens"
    assert last_result.output_tokens > 0, "Should have output tokens"
    assert last_result.duration_seconds > 0, "Should have duration"


@pytest.mark.asyncio
async def test_reasoning_agent_with_multiple_tools(calculator_tool, weather_tool, tool_executors):
    """Test the ReasoningAgent with multiple tools using GPT-4o-mini."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[calculator_tool, weather_tool],
        tool_executors=tool_executors,
        max_iterations=3,
        temperature=0
    )
    
    # Test messages with a complex query requiring multiple tools
    messages = [{
        "role": "user", 
        "content": "I'm planning a trip to New York. What's the weather like there? Also, if the temperature in Celsius is 22, what is that in Fahrenheit?"
    }]
    
    # Run the agent and collect the results
    results = []
    async for result in agent.run_async(messages):
        results.append(result)
    
    # Check that both tools were used
    tool_predictions = [
        r for r in results 
        if isinstance(r, ResponseChunk) and r.content_type == ContentType.TOOL_PREDICTION
    ]
    
    tool_names = [p.content for p in tool_predictions]
    tool_names_str = " ".join(tool_names)
    
    # At least one of the tools should be used
    assert "calculator" in tool_names_str or "get_weather" in tool_names_str, "Should use at least one tool"
    
    # Last result should be a ResponseSummary with a complete answer
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"
    
    # The answer should mention weather and temperature conversion
    final_answer = last_result.content
    assert "New York" in final_answer, "Answer should mention New York"
    assert "Fahrenheit" in final_answer, "Answer should mention Fahrenheit"
    
    # Check token accounting
    assert last_result.input_tokens > 0, "Should have input tokens"
    assert last_result.output_tokens > 0, "Should have output tokens"


@pytest.mark.asyncio
async def test_reasoning_agent_no_tools_needed(tool_executors):
    """Test the ReasoningAgent with a question that doesn't need tools."""
    # Create the reasoning agent
    agent = ReasoningAgent(
        model_name="gpt-4o-mini",
        tools=[],  # No tools
        tool_executors=tool_executors,
        max_iterations=2,
        temperature=0
    )
    
    # Test messages with a question that doesn't need tools
    messages = [{
        "role": "user", 
        "content": "What are three benefits of regular exercise?"
    }]
    
    # Run the agent and collect the results
    results = []
    async for result in agent.run_async(messages):
        results.append(result)
    
    # Check thinking occurred
    thinking_chunks = [
        r for r in results 
        if isinstance(r, ResponseChunk) and r.content_type == ContentType.THINKING
    ]
    assert len(thinking_chunks) > 0, "Should have thinking chunks"
    
    # Last result should be a ResponseSummary with a complete answer
    last_result = results[-1]
    assert isinstance(last_result, ResponseSummary), "Last result should be ResponseSummary"
    
    # The answer should mention benefits of exercise
    final_answer = last_result.content
    exercise_terms = ["exercise", "health", "fitness", "cardiovascular", "strength", "mental"]
    assert any(term in final_answer.lower() for term in exercise_terms), "Answer should discuss exercise benefits"
    
    # There should be no tool predictions
    tool_predictions = [
        r for r in results 
        if isinstance(r, ResponseChunk) and r.content_type == ContentType.TOOL_PREDICTION
    ]
    assert len(tool_predictions) == 0, "Should not have tool predictions"