"""Reasoning agent module with structured output and event-driven flow."""

import asyncio
import json
from enum import Enum, auto
from textwrap import dedent
from typing import AsyncGenerator, Callable, Dict, List, Literal, Optional, Union, Any

from pydantic import BaseModel, Field

from sik_llms.models_base import (
    Client, RegisteredClients, ResponseChunk, ResponseSummary, Tool, ToolChoice,
    ContentType, ToolPrediction, ToolPredictionResponse, BaseModel as SIKBaseModel, system_message
)
from sik_llms.openai import (
    CHAT_MODEL_COST_PER_TOKEN as OPENAI_CHAT_MODEL_COST_PER_TOKEN,
)
from sik_llms.anthropic import (
    CHAT_MODEL_COST_PER_TOKEN as ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN,
)


class ReasoningAction(str, Enum):
    """Possible actions the reasoning agent can take."""
    CONTINUE_THINKING = "continue_thinking"
    USE_TOOL = "use_tool"
    PROVIDE_ANSWER = "provide_answer"


class ReasoningStep(BaseModel):
    """Model for a single reasoning step."""
    thought: str = Field(description="Current reasoning/thinking about the problem")
    is_complete: bool = Field(description="Whether the reasoning process is complete")
    next_action: ReasoningAction = Field(description="What action to take next")
    tool_name: str | None = Field(default=None, description="Name of the tool to use (if next_action is USE_TOOL)")


class ReasoningState(Enum):
    """Internal states for the reasoning agent."""
    THINKING = auto()
    USING_TOOL = auto()
    COMPLETE = auto()


@Client.register(RegisteredClients.REASONING_AGENT)
class ReasoningAgent(Client):
    """
    A reasoning agent that can iteratively think and execute tools using structured output.
    
    This implementation uses a Pydantic model for structured reasoning steps and 
    an event-driven approach to provide progress updates to the client.
    """

    def __init__(
            self,
            model_name: str,
            tools: Optional[List[Tool]] = None,
            tool_choice: ToolChoice = ToolChoice.AUTO,
            max_iterations: int = 5,
            tool_executors: Optional[Dict[str, Callable]] = None,
            reasoning_system_prompt: Optional[str] = None,
            **model_kwargs: dict,
        ):
        """
        Initialize the reasoning agent.
        
        Args:
            model_name: The name of the model to use
            tools: Optional list of tools the agent can use
            tool_choice: Whether tool use is required or auto
            max_iterations: Maximum number of iterations for thinking and tool use
            tool_executors: Dict mapping tool names to functions that execute them
            reasoning_system_prompt: Custom system prompt for the reasoning process
            **model_kwargs: Additional arguments to pass to the model
        """
        # Determine which client types to use based on the model name
        if model_name in OPENAI_CHAT_MODEL_COST_PER_TOKEN:
            self.client_type = RegisteredClients.OPENAI
            self.tools_client_type = RegisteredClients.OPENAI_TOOLS
        elif model_name in ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN:
            self.client_type = RegisteredClients.ANTHROPIC
            self.tools_client_type = RegisteredClients.ANTHROPIC_TOOLS
        else:
            raise ValueError(f"Unknown model name '{model_name}'")

        # Initialize the thinking model
        self.model = model_name
        self.model_kwargs = model_kwargs.copy()
        
        # Create the reasoning model with structured output
        self.reasoning_model = Client.instantiate(
            client_type=self.client_type,
            model_name=model_name,
            response_format=ReasoningStep,
            **model_kwargs,
        )
        
        # Configure tools and tool execution
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.tool_executors = tool_executors or {}
        self.max_iterations = max_iterations
        
        # Cache for tool clients
        self.tools_clients = {}
        
        # Set up the system prompt for reasoning
        self.reasoning_system_prompt = reasoning_system_prompt or self._default_reasoning_prompt()

        # Track the total usage stats
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_cost = 0
        self.total_output_cost = 0
        self.start_time = None

    def _default_reasoning_prompt(self) -> str:
        """Default system prompt for reasoning."""
        tools_description = ""
        if self.tools:
            tools_description = "Available tools:\n"
            for tool in self.tools:
                tools_description += f"- {tool.name}: {tool.description or 'No description provided'}\n"
                if tool.parameters:
                    tools_description += "  Parameters:\n"
                    for param in tool.parameters:
                        required_str = "(required)" if param.required else "(optional)"
                        tools_description += f"  - {param.name} {required_str}: {param.description or 'No description'}\n"

        return dedent(f"""
        You are a reasoning agent that solves problems step-by-step.
        
        PROCESS:
        1. Think carefully about the user's request or question
        2. Break down complex problems into smaller steps
        3. When needed, use available tools to gather information or perform actions
        4. After each step, determine if you have enough information to provide a final answer
        5. If you need more information, continue reasoning and using tools
        6. When you have enough information, set is_complete to true
        
        {tools_description}
        
        For each step, you must provide:
        1. Your current thought process (thought)
        2. Whether reasoning is complete (is_complete)
        3. The next action to take (next_action): continue_thinking, use_tool, or provide_answer
        4. If using a tool, provide ONLY the tool name - do not provide tool parameters
        
        Always take your time to think and reason through the problem carefully.
        """).strip()

    def _get_tools_client(self, tool_name: str) -> Client:
        """Get or create a tools client for the given tool."""
        # If a specific tool is requested, filter to just that tool
        tools_to_use = [tool for tool in self.tools if tool_name is None or tool.name == tool_name]
        if not tools_to_use:
            raise ValueError(f"Tool '{tool_name}' not found")

        return Client.instantiate(
            client_type=self.tools_client_type,
            model_name=self.model,
            tools=tools_to_use,
            tool_choice=ToolChoice.REQUIRED if tool_name else self.tool_choice,
            **self.model_kwargs,
        )

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool with the given arguments.
        
        Handles both synchronous and asynchronous tool executor functions.
        """
        executor = self.tool_executors.get(tool_name)
        if not executor:
            raise ValueError(f"No executor available for tool '{tool_name}'")
        
        # Check if the executor is a coroutine function
        if asyncio.iscoroutinefunction(executor):
            return await executor(**args)
        else:
            # Run synchronous functions in a thread pool to avoid blocking
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: executor(**args)
            )

    async def run_async(
            self,
            messages: List[Dict[str, Any]],
        ) -> AsyncGenerator[ResponseChunk | ResponseSummary, None]:
        """
        Run the reasoning agent on the given messages.

        This method implements an iterative reasoning process using structured output.
        """
        self.start_time = asyncio.get_event_loop().time()
        # Add the system message for reasoning
        reasoning_messages = [
            system_message(self.reasoning_system_prompt),
            *messages.copy(),
        ]
        # Keep track of all reasoning steps and tool calls
        reasoning_history = []
        final_reasoning = []
        iteration = 0
        state = ReasoningState.THINKING

        yield ResponseChunk(
            content='',
            content_type=ContentType.THINKING,
            iteration=iteration,
        )

        while iteration < self.max_iterations and state != ReasoningState.COMPLETE:
            iteration += 1
            if state == ReasoningState.THINKING:
                # Get structured reasoning step
                response = self.reasoning_model(reasoning_messages)
                # Update token usage
                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                self.total_input_cost += response.input_cost
                self.total_output_cost += response.output_cost

                # Parse the reasoning step
                reasoning_step = response.content.parsed

                if not reasoning_step:
                    # Handle parsing failure
                    yield ResponseChunk(
                        content=f"Error: Failed to parse reasoning step. Response: {response.content.refusal}",  # noqa: E501
                        content_type=ContentType.ERROR,
                        iteration=iteration,
                    )
                    break

                # Emit thinking event
                yield ResponseChunk(
                    content=reasoning_step.thought,
                    content_type=ContentType.THINKING,
                    iteration=iteration,
                )

                # Record the reasoning step
                reasoning_history.append({
                    'iteration': iteration,
                    'reasoning_step': reasoning_step.model_dump(),
                })
                final_reasoning.append(reasoning_step.thought)
                
                # Determine next state based on the reasoning step
                if reasoning_step.is_complete:
                    state = ReasoningState.COMPLETE
                    # yield ResponseChunk(
                    #     content="Reasoning complete. Generating final answer...",
                    #     content_type=ContentType.INFO,
                    #     iteration=iteration,
                    # )
                elif reasoning_step.next_action == ReasoningAction.USE_TOOL and self.tools:
                    # Validate tool selection
                    if not reasoning_step.tool_name:
                        yield ResponseChunk(
                            content="Error: Tool name is missing",
                            content_type=ContentType.ERROR,
                            iteration=iteration
                        )
                        continue
                    
                    # Move to tool use state
                    state = ReasoningState.USING_TOOL
                    tool_name = reasoning_step.tool_name
                    
                    # Use the tools client to predict the tool inputs
                    try:
                        # Create messages for tool prediction
                        tool_messages = reasoning_messages.copy()
                        tool_messages.append({
                            'role': 'assistant', 
                            'content': reasoning_step.thought
                        })
                        
                        # Get the tools client for this specific tool
                        tools_client = self._get_tools_client(tool_name)
                        tool_response = tools_client(tool_messages)
                        
                        # Update token usage
                        self.total_input_tokens += tool_response.input_tokens
                        self.total_output_tokens += tool_response.output_tokens
                        self.total_input_cost += tool_response.input_cost
                        self.total_output_cost += tool_response.output_cost
                        
                        if tool_response.tool_prediction:
                            # Use the tool name and arguments from the prediction
                            predicted_tool_name = tool_response.tool_prediction.name
                            tool_input = tool_response.tool_prediction.arguments
                            
                            # Verify the predicted tool matches the requested tool
                            if predicted_tool_name != tool_name:
                                yield ResponseChunk(
                                    content=f"Warning: Tool prediction changed from {tool_name} to {predicted_tool_name}",
                                    content_type=ContentType.TEXT,
                                    iteration=iteration
                                )
                                tool_name = predicted_tool_name
                        else:
                            # No tool prediction was made
                            yield ResponseChunk(
                                content=f"Error: Failed to get tool inputs for {tool_name}",
                                content_type=ContentType.ERROR,
                                iteration=iteration
                            )
                            # Fall back to thinking state
                            state = ReasoningState.THINKING
                            continue
                        
                        # Yield tool prediction event
                        yield ResponseChunk(
                            content=f"Using tool: {tool_name} with parameters: {json.dumps(tool_input, indent=2)}",
                            content_type=ContentType.TOOL_PREDICTION,
                            iteration=iteration
                        )
                        
                        # Execute the tool
                        try:
                            tool_result = await self._execute_tool(tool_name, tool_input)
                            
                            # Yield tool result event
                            yield ResponseChunk(
                                content=f"Tool result: {tool_result}",
                                content_type=ContentType.TOOL_RESULT,
                                iteration=iteration
                            )
                            
                            # Add tool use to the messages
                            reasoning_messages.append({
                                'role': 'assistant',
                                'content': reasoning_step.thought
                            })
                            reasoning_messages.append({
                                'role': 'user',
                                'content': f"Tool: {tool_name}\nParameters: {json.dumps(tool_input)}\nResult: {tool_result}\n\nContinue your reasoning based on this tool result."
                            })
                            
                            # Record the tool use
                            reasoning_history.append({
                                'iteration': iteration,
                                'tool_use': {
                                    'name': tool_name,
                                    'input': tool_input,
                                    'result': tool_result
                                }
                            })
                            
                            # Return to thinking state
                            state = ReasoningState.THINKING
                        except Exception as e:
                            # Handle tool execution errors
                            error_message = f"Error executing tool {tool_name}: {str(e)}"
                            yield ResponseChunk(
                                content=error_message,
                                content_type=ContentType.ERROR,
                                iteration=iteration
                            )
                            
                            # Add error to messages
                            reasoning_messages.append({
                                'role': 'assistant',
                                'content': reasoning_step.thought
                            })
                            reasoning_messages.append({
                                'role': 'user',
                                'content': f"{error_message}\n\nContinue your reasoning without using this tool."
                            })
                            
                            # Return to thinking state
                            state = ReasoningState.THINKING
                    except Exception as e:
                        # Handle errors in tool prediction
                        error_message = f"Error predicting tool inputs for {tool_name}: {str(e)}"
                        yield ResponseChunk(
                            content=error_message,
                            content_type=ContentType.ERROR,
                            iteration=iteration
                        )
                        
                        # Add error to messages and continue reasoning
                        reasoning_messages.append({
                            'role': 'assistant',
                            'content': reasoning_step.thought
                        })
                        reasoning_messages.append({
                            'role': 'user',
                            'content': f"{error_message}\n\nContinue your reasoning without using this tool."
                        })
                        
                        # Return to thinking state
                        state = ReasoningState.THINKING
                
                # Continue thinking
                elif reasoning_step.next_action == ReasoningAction.CONTINUE_THINKING:
                    reasoning_messages.append({
                        'role': 'assistant',
                        'content': reasoning_step.thought
                    })
                    reasoning_messages.append({
                        'role': 'user',
                        'content': "Continue your reasoning."
                    })
                
                # Provide answer
                elif reasoning_step.next_action == ReasoningAction.PROVIDE_ANSWER:
                    state = ReasoningState.COMPLETE
                    yield ResponseChunk(
                        content="Reasoning complete. Generating final answer...",
                        content_type=ContentType.TEXT,
                        iteration=iteration
                    )
        
        # Check if we hit max iterations
        if iteration >= self.max_iterations and state != ReasoningState.COMPLETE:
            yield ResponseChunk(
                content=f"Maximum iterations ({self.max_iterations}) reached. Generating best answer with current information.",
                content_type=ContentType.ERROR,
                iteration=iteration,
            )
        
        # Generate the final streaming response using the regular model (not structured output)
        # Create a new model instance without structured output for streaming
        final_model = Client.instantiate(
            client_type=self.client_type,
            model_name=self.model,
            **self.model_kwargs,
        )
        
        # Prepare the final prompt with all the reasoning history
        final_messages = messages.copy()
        
        # Add a system message with the reasoning history
        reasoning_summary = "\n\n".join([
            f"Step {i+1}: {reasoning}" for i, reasoning in enumerate(final_reasoning)
        ])
        
        tool_summary = ""
        if any("tool_use" in step for step in reasoning_history):
            tool_uses = [step["tool_use"] for step in reasoning_history if "tool_use" in step]
            tool_summary = "\n\n".join([
                f"Tool: {tool['name']}\nParameters: {json.dumps(tool['input'])}\nResult: {tool['result']}"
                for tool in tool_uses
            ])
        
        final_system_message = {
            'role': 'system',
            'content': dedent(f"""
            You are providing a final answer based on a reasoning process.
            
            Reasoning process:

            {reasoning_summary}
            
            {f"Tool usage:\n{tool_summary}" if tool_summary else ""}
            
            Use this information to provide a final answer to the user's question. The answer should be complete and accurate and summarize how the conclusion was made.
            """).strip()
        }
        
        final_answer = ""
        async for chunk in final_model.run_async([*messages, final_system_message]):
            if isinstance(chunk, ResponseChunk):
                if chunk.content:
                    final_answer += chunk.content
                    yield ResponseChunk(
                        content=chunk.content,
                        content_type=ContentType.TEXT
                    )
            elif isinstance(chunk, ResponseSummary):
                # Update token usage stats
                self.total_input_tokens += chunk.input_tokens
                self.total_output_tokens += chunk.output_tokens
                self.total_input_cost += chunk.input_cost
                self.total_output_cost += chunk.output_cost
        
        # Calculate total duration
        end_time = asyncio.get_event_loop().time()
        duration = end_time - self.start_time
        
        # Build the full response
        complete_response = final_answer
        
        # Yield the final summary
        yield ResponseSummary(
            content=complete_response,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
            input_cost=self.total_input_cost,
            output_cost=self.total_output_cost,
            duration_seconds=duration
        )