"""
Tests for the MCPClientManager class and ensures that the tools and Tool objects are correctly
loaded and called.
"""
import os
import re
import pytest
from mcp.types import CallToolResult, TextContent
from sik_llms.mcp_manager import MCPClientManager
from sik_llms.models_base import (
    ErrorEvent,
    TextResponse,
    ThinkingEvent,
    ToolPredictionEvent,
    ToolResultEvent,
    user_message,
)
from sik_llms.reasoning_agent import ReasoningAgent
from tests.conftest import ANTRHOPIC_TEST_THINKING_MODEL, OPENAI_TEST_MODEL


@pytest.mark.asyncio
async def test_mcp_manager(mcp_fake_server_config: dict):
    async with MCPClientManager(mcp_fake_server_config) as manager:
        tools = manager.get_tool_infos()
        expected_tools = [
            'reverse_text',
            'calculator_sum',
            'calculator',
            'count_characters',
            'get_weather',
        ]
        assert {tool['name'] for tool in tools} == set(expected_tools)
        assert {tool.name for tool in manager.get_tools()} == set(expected_tools)
        expected_servers = {'fake-server-text', 'fake-server-calculator', 'fake-server-misc'}
        assert {tool['server'] for tool in tools} == expected_servers
        assert all(tool['tool'] for tool in tools)
        ####
        # Check tool names
        ####
        assert manager.get_tool('reverse_text').name == 'reverse_text'
        assert manager.get_tool('calculator_sum').name == 'calculator_sum'
        assert manager.get_tool('calculator').name == 'calculator'
        assert manager.get_tool('count_characters').name == 'count_characters'
        assert manager.get_tool('get_weather').name == 'get_weather'
        ####
        # Check tool descriptions
        ####
        assert 'Reverse the input text.'in manager.get_tool('reverse_text').description
        assert 'Calculate the sum of a list of numbers.' in manager.get_tool('calculator_sum').description  # noqa: E501
        assert 'Calculate the expresion.' in manager.get_tool('calculator').description
        assert 'Count the number of words in a text.' in manager.get_tool('count_characters').description  # noqa: E501
        assert 'Get the weather for a location.' in manager.get_tool('get_weather').description
        ####
        # Check tool parameters
        ####
        assert manager.get_tool('reverse_text').parameters[0].name == 'text'
        assert manager.get_tool('reverse_text').parameters[0].param_type is str
        assert manager.get_tool('reverse_text').parameters[0].required is True
        assert manager.get_tool('reverse_text').parameters[0].valid_values is None
        assert manager.get_tool('reverse_text').parameters[0].any_of is None

        assert manager.get_tool('calculator_sum').parameters[0].name == 'numbers'
        assert manager.get_tool('calculator_sum').parameters[0].param_type is list
        assert manager.get_tool('calculator_sum').parameters[0].required is True
        assert manager.get_tool('calculator_sum').parameters[0].valid_values is None
        assert manager.get_tool('calculator_sum').parameters[0].any_of is None

        assert manager.get_tool('get_weather').parameters[0].name == 'location'
        assert manager.get_tool('get_weather').parameters[0].param_type is str
        assert manager.get_tool('get_weather').parameters[0].required is True
        assert manager.get_tool('get_weather').parameters[0].valid_values is None
        assert manager.get_tool('get_weather').parameters[0].any_of is None

        assert manager.get_tool('get_weather').parameters[1].name == 'units'
        assert manager.get_tool('get_weather').parameters[1].param_type is str
        assert manager.get_tool('get_weather').parameters[1].required is True
        assert set(manager.get_tool('get_weather').parameters[1].valid_values) == {'celsius', 'fahrenheit'}  # noqa: E501
        assert manager.get_tool('get_weather').parameters[1].any_of is None
        assert manager.get_tool('get_weather').parameters[1].description


@pytest.mark.asyncio
async def test_mcp_tool_calling__async(mcp_fake_server_config: dict):
    async with MCPClientManager(mcp_fake_server_config) as manager:
        ####
        # Calling the tool from the .func method of Tool object should behave the same as calling
        # the underlying function directly; however, calling the tool from the .call_tool method
        # should return an mcp CallToolResult object.
        ####
        args = {'text': 'asdf'}

        tool = manager.get_tool('reverse_text')
        result = await tool.func(**args)
        assert result == 'fdsa'

        result_tool_call = await manager.call_tool('reverse_text', args)
        assert isinstance(result_tool_call, CallToolResult)
        assert isinstance(result_tool_call.content[0], TextContent)
        assert result_tool_call.isError is False
        assert result_tool_call.content[0].text == 'fdsa'

@pytest.mark.asyncio
async def test_mcp_tool_calling__sync(mcp_fake_server_config: dict):
    async with MCPClientManager(mcp_fake_server_config) as manager:
        ####
        # calculator in mcp_fake_server_calculator.py is a sync function
        ####
        args = {'expression': '2 + 2'}

        tool = manager.get_tool('calculator')
        result = await tool.func(**args)
        assert result == '4'

        result_tool_call = await manager.call_tool('calculator', args)
        assert isinstance(result_tool_call, CallToolResult)
        assert isinstance(result_tool_call.content[0], TextContent)
        assert result_tool_call.isError is False
        assert result_tool_call.content[0].text == '4'


@pytest.mark.asyncio
async def test_mcp_tool_calling__error_async(mcp_error_server_config: dict):
    async with MCPClientManager(mcp_error_server_config) as manager:
        ####
        # calculator in mcp_fake_server_calculator.py is a sync function
        ####
        args = {'text': 'this is text'}
        tool = manager.get_tool('error_async')
        with pytest.raises(Exception, match="An error occurred"):
            _ = await tool.func(**args)

        result_tool_call = await manager.call_tool('error_async', args)
        assert isinstance(result_tool_call, CallToolResult)
        assert isinstance(result_tool_call.content[0], TextContent)
        assert result_tool_call.isError is True
        assert "An error occurred" in result_tool_call.content[0].text


@pytest.mark.asyncio
async def test_mcp_tool_calling__error_sync(mcp_error_server_config: dict):
    async with MCPClientManager(mcp_error_server_config) as manager:
        ####
        # calculator in mcp_fake_server_calculator.py is a sync function
        ####
        args = {'text': 'this is text'}
        tool = manager.get_tool('error_sync')
        with pytest.raises(Exception, match="An error occurred"):
            _ = await tool.func(**args)

        result_tool_call = await manager.call_tool('error_async', args)
        assert isinstance(result_tool_call, CallToolResult)
        assert isinstance(result_tool_call.content[0], TextContent)
        assert result_tool_call.isError is True
        assert "An error occurred" in result_tool_call.content[0].text


@pytest.mark.asyncio
async def test_reasoning_agent_with_mcp__test_prompt(
        mcp_fake_server_config: dict,
        test_files_path: str,
    ):
    async with MCPClientManager(mcp_fake_server_config) as manager:
        agent = ReasoningAgent(
            model_name=OPENAI_TEST_MODEL,
            tools=manager.get_tools(),
        )
        prompt = agent._create_reasoning_prompt()
        with open(f'{test_files_path}/reasoning_prompt_mcp_tools.txt', 'w') as f:  # noqa: ASYNC230
            f.write(prompt)
        expected_tools = [
            'reverse_text',
            'calculator_sum',
            'calculator',
            'count_characters',
            'get_weather',
        ]
        for tool in expected_tools:
            assert tool in prompt


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.stochastic(samples=5, threshold=0.5)
@pytest.mark.parametrize('model_name', [
    pytest.param(
        OPENAI_TEST_MODEL,
        id="OpenAI",
    ),
    pytest.param(
        ANTRHOPIC_TEST_THINKING_MODEL,
        id="Anthropic",
        marks=pytest.mark.skipif(
            os.getenv('ANTHROPIC_API_KEY') is None,
            reason="ANTHROPIC_API_KEY is not set",
        ),
    ),
])
async def test_reasoning_agent_with_calculator(mcp_fake_server_config: dict, model_name: str):
    """Test the ReasoningAgent with a calculator tool using GPT-4o-mini."""
    async with MCPClientManager(mcp_fake_server_config) as manager:
        agent = ReasoningAgent(
            model_name=model_name,
            tools=manager.get_tools(),
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
@pytest.mark.stochastic(samples=5, threshold=0.5)
async def test_reasoning_agent__tool_raises_exception(mcp_error_server_config: dict):
    """Test the ReasoningAgent with a calculator tool using GPT-4o-mini."""
    async with MCPClientManager(mcp_error_server_config) as manager:
        agent = ReasoningAgent(
            model_name=OPENAI_TEST_MODEL,
            tools=manager.get_tools(),
            max_iterations=2,
            temperature=0,
        )

        results = []
        async for result in agent.stream([user_message("This is a test. Call the `error_async` tool and pass in 'test' as the `text` parameter.")]):  # noqa: E501
            results.append(result)

        # Check that we got the expected results
        thinking_events = [r for r in results if isinstance(r, ThinkingEvent)]
        assert len(thinking_events) > 0, "Should have thinking events"

        error_events = [r for r in results if isinstance(r, ErrorEvent)]
        assert len(error_events) > 0, "Should have errors"
        assert "Error executing tool" in error_events[0].content

        tool_prediction_events = [r for r in results if isinstance(r, ToolPredictionEvent)]
        assert len(tool_prediction_events) > 0, "Should have tool prediction events"
        assert tool_prediction_events[0].name == "error_async"

        tool_result_events = [r for r in results if isinstance(r, ToolResultEvent)]
        assert len(tool_result_events) == 0, "Should not have tool result events since the tool raised an exception"  # noqa: E501

        # Last result should be a TextResponse with summary
        last_result = results[-1]
        assert last_result.input_tokens > 0, "Should have input tokens"
        assert last_result.output_tokens > 0, "Should have output tokens"
        assert last_result.duration_seconds > 0, "Should have duration"


