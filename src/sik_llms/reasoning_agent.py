"""Reasoning agent module with structured output and event-driven flow."""

import asyncio
import json
from enum import Enum
from textwrap import dedent
from typing import AsyncGenerator, Callable, Dict, Optional, Any  # noqa: UP035
from pydantic import BaseModel, Field
from sik_llms.models_base import (
    Client,
    ErrorEvent,
    RegisteredClients,
    TextChunkEvent,
    ResponseSummary,
    ThinkingEvent,
    Tool,
    ToolChoice,
    ToolPredictionEvent,
    ToolResultEvent,
    system_message,
    assistant_message,
    user_message,
)
from sik_llms.openai import (
    CHAT_MODEL_COST_PER_TOKEN as OPENAI_CHAT_MODEL_COST_PER_TOKEN,
)
from sik_llms.anthropic import (
    CHAT_MODEL_COST_PER_TOKEN as ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN,
)


class ReasoningAction(str, Enum):
    """Possible actions the reasoning agent can take."""

    CONTINUE_THINKING = 'continue_thinking'
    USE_TOOL = 'use_tool'
    FINISHED = 'finished'


class ReasoningStep(BaseModel):
    """Model for a single reasoning step."""

    thought: str = Field(description="Current reasoning/thinking about the problem")
    next_action: ReasoningAction = Field(description="What action to take next")
    tool_name: str | None = Field(default=None, description="Name of the tool to use (if next_action is USE_TOOL)")  # noqa: E501


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
            tools: Optional[list[Tool]] = None,
            max_iterations: int = 5,
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
        if any(t.func is None for t in tools):
            raise ValueError("All tools must have a callable function")
        self.tools = tools or []
        self.max_iterations = max_iterations
        # Set up the system prompt for reasoning
        self.reasoning_system_prompt = reasoning_system_prompt or self._default_reasoning_prompt()


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

        [PROCESS]:

        1. Think carefully about the user's request or question
        2. Break down complex problems into smaller steps
        3. When needed, use available tools (if available) to gather information or perform actions
        4. After each step, determine if you have enough information to provide a final answer
        5. Continue to reason and use tools (if available) until you can provide a complete answer.
        6. When you have enough information, set `next_action` to `FINISHED`
        7. If you need additional information from the user in order to answer the question, specify the information you need and set `next_action` to `FINISHED`. This will end the reasoning process so the user can provide the necessary information. This prevents hallucinations.

        {tools_description}

        For each step, you must provide:
        1. Your current thought process (`thought`)
        2. The next action to take (`next_action`): `CONTINUE_THINKING`, `USE_TOOL`, or `FINISHED`
        3. If using a tool, provide ONLY the tool name and provide the EXACT name - DO NOT PROVIDE TOOL PARAMETERS

        Always take your time to think and reason through the problem carefully.
        """).strip()  # noqa: E501

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
            tool_choice=ToolChoice.REQUIRED,
            **self.model_kwargs,
        )

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool with the given arguments.

        Handles both synchronous and asynchronous tool executor functions.
        """
        executor = next((tool.func for tool in self.tools if tool.name == tool_name), None)
        if not executor:
            raise ValueError(f"No executor available for tool '{tool_name}'")

        # Check if the executor is a coroutine function
        if asyncio.iscoroutinefunction(executor):
            return await executor(**args)

        # Run synchronous functions in a thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: executor(**args)
        )

    async def run_async(  # noqa: PLR0912, PLR0915
            self,
            messages: list[dict[str, Any]],
        ) -> AsyncGenerator[
            TextChunkEvent | ThinkingEvent | ToolPredictionEvent | ToolResultEvent
            | ErrorEvent | ResponseSummary,
        ]:
        """
        Run the reasoning agent on the given messages.

        This method implements an iterative reasoning process using structured output.
        """
        start_time = asyncio.get_event_loop().time()
        # Track the total usage stats
        total_input_tokens = 0
        total_output_tokens = 0
        total_input_cost = 0
        total_output_cost = 0
        # Get the last message from the user; treat the previous messages as text/context
        if len(messages) == 0:
            raise ValueError("No messages provided.")

        messages = messages.copy()
        last_message = messages.pop()
        # reasoning_messages will be used to send to the various models/agents
        reasoning_messages = [
            # Add the system message for reasoning
            system_message(self.reasoning_system_prompt),
        ]
        if messages:
            # if there were more than one message, add them to the reasoning messages
            reasoning_messages.append(user_message(f"Here are the previous messages for context:\n\n```\n{json.dumps(messages)}\n```\n"))  # noqa: E501
        reasoning_messages.append(last_message)
        # reasoning_history will be used to store the reasoning steps so that we can summarize them
        # later for the final response
        reasoning_history: list[dict] = []
        iteration = 0
        reasoning_step: ReasoningStep | None = None

        # let's the user know that the agent has started thinking
        yield ThinkingEvent(
            content='',
            iteration=iteration,
        )

        while iteration < self.max_iterations:
            iteration += 1
            # Get structured reasoning step

            response: ResponseSummary = self.reasoning_model(reasoning_messages)
            # Update token usage
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_input_cost += response.input_cost
            total_output_cost += response.output_cost

            # Parse the reasoning step
            reasoning_step: ReasoningStep = response.response.parsed

            # check if the structured response was successfully parsed/created
            if reasoning_step:
                reasoning_messages.append(assistant_message(reasoning_step.thought))
                reasoning_history.append({
                    'iteration': iteration,
                    'reasoning_step': reasoning_step.model_dump(),
                })
            else:
                # Handle parsing failure
                error_message = f"Error: Failed to parse reasoning step. Response: {response.response.refusal}"  # noqa: E501
                yield ErrorEvent(
                    content=error_message,
                    metadata={
                        'response': response,
                        'iteration': iteration,
                    },
                )
                reasoning_messages.extend([
                    assistant_message(error_message),
                    user_message("Try again and adjust your thinking or response based on the error message."),  # noqa: E501
                ])
                continue

            # Emit thinking event
            yield ThinkingEvent(
                content=reasoning_step.thought,
                iteration=iteration,
            )

            # Determine next step based on the reasoning step
            if reasoning_step.next_action == ReasoningAction.FINISHED:
                # Finish the reasoning process
                break
            # Continue thinking
            elif reasoning_step.next_action == ReasoningAction.CONTINUE_THINKING:
                # TODO perhaps add a user message to encourage the model to continue thinking?
                continue
            # Use a tool
            elif reasoning_step.next_action == ReasoningAction.USE_TOOL:
                if not self.tools:
                    yield ErrorEvent(
                        content="Error: Agent chose to use tools but no tools are available.",
                        metadata={
                            'reasoning_step': reasoning_step,
                            'iteration': iteration,
                        },
                    )
                    reasoning_messages.extend([
                        assistant_message("I chose to use a tool but no tools are available."),
                        user_message("Continue your reasoning without using tools."),
                    ])
                    continue
                # Validate tool selection
                if not reasoning_step.tool_name:
                    yield ErrorEvent(
                        content="Error: Tool name is missing",
                        metadata={
                            'reasoning_step': reasoning_step,
                            'iteration': iteration,
                        },
                    )
                    reasoning_messages.extend([
                        assistant_message("Error: I chose to use a tool but didn't provide the tool name."),  # noqa: E501
                        user_message("Adjust your response and use the correct tools available if they are needed to answer the question."),  # noqa: E501
                    ])
                    continue

                tool_name = reasoning_step.tool_name

                # Use the tools client to predict the tool inputs
                try:
                    # Create messages for tool prediction
                    tool_messages = reasoning_messages.copy()
                    # remove initial system message that is specific to the reasoning agent
                    tool_messages.pop(0)
                    tool_messages[-1]['content'] += f"\n\nI will use the `{tool_name}` tool to help solve the problem."  # noqa: E501

                    # Get the tools client for this specific tool
                    tools_client = self._get_tools_client(tool_name)
                    tool_response = tools_client(tool_messages)
                    # Update token usage
                    total_input_tokens += tool_response.input_tokens
                    total_output_tokens += tool_response.output_tokens
                    total_input_cost += tool_response.input_cost
                    total_output_cost += tool_response.output_cost

                    if tool_response.tool_prediction:
                        # Use the tool name and arguments from the prediction
                        predicted_tool_name = tool_response.tool_prediction.name
                        predicted_tool_args = tool_response.tool_prediction.arguments

                        # Verify the predicted tool matches the requested tool
                        if predicted_tool_name != tool_name:
                            yield ErrorEvent(
                                content=f"Warning: Tool prediction changed from `{tool_name}` to `{predicted_tool_name}`",  # noqa: E501
                                metadata={
                                    'tool_name': tool_name,
                                    'predicted_tool_name': predicted_tool_name,
                                    'tool_input': predicted_tool_args,
                                    'iteration': iteration,
                                },
                            )
                            tool_name = predicted_tool_name

                        # Yield tool prediction event
                        yield ToolPredictionEvent(
                            name=predicted_tool_name,
                            arguments=predicted_tool_args,
                            iteration=iteration,
                        )

                        # Execute the tool
                        try:
                            tool_result = await self._execute_tool(predicted_tool_name, predicted_tool_args)  # noqa: E501
                            # Yield tool result event
                            yield ToolResultEvent(
                                name=predicted_tool_name,
                                arguments=predicted_tool_args,
                                result=tool_result,
                                iteration=iteration,
                            )
                            # Add tool use to the messages
                            reasoning_messages.extend([
                                assistant_message(f"Tool Used: `{tool_name}`\n\nParameters: `{json.dumps(predicted_tool_args)}`\n\nResult:\n\n```\n{tool_result!s}\n```\n"),  # noqa: E501
                                user_message("Continue your reasoning based on the tool result."),
                            ])
                            # Record the tool use
                            reasoning_history.append({
                                'iteration': iteration,
                                'tool_use': {
                                    'name': tool_name,
                                    'input': predicted_tool_args,
                                    'result': str(tool_result),
                                },
                            })
                        except Exception as e:
                            # Handle tool execution errors
                            error_message = f"Error executing tool {tool_name}: {e!s}"
                            yield ErrorEvent(
                                content=error_message,
                                metadata={
                                    'tool_name': tool_name,
                                    'tool_input': predicted_tool_args,
                                    'error': str(e),
                                    'iteration': iteration,
                                },
                            )
                            reasoning_messages.extend([
                                assistant_message(error_message),
                                user_message("Either adjust your response based on the error, or continue your reasoning without using this tool."),  # noqa: E501
                            ])
                    else:
                        # No tool prediction was made
                        response_message = tool_response.message
                        # No tool prediction was made
                        error_message = f"Error: Failed to get tool prediction for {tool_name}; message=`{response_message}`"  # noqa: E501
                        yield ErrorEvent(
                            content=error_message,
                            metadata={
                                'tool_name': tool_name,
                                'message': response_message,
                                'iteration': iteration,
                            },
                        )
                        reasoning_messages.extend([
                            assistant_message(error_message),
                            user_message("Either adjust your response based on the error, or continue your reasoning without using this tool."),  # noqa: E501
                        ])
                        continue

                except Exception as e:
                    # Handle errors in tool prediction
                    error_message = f"Error predicting tool inputs for {tool_name}: {e!s}"
                    yield ErrorEvent(
                        content=error_message,
                        metadata={
                            'tool_name': tool_name,
                            'error': str(e),
                            'iteration': iteration,
                        },
                    )
                    reasoning_messages.append([
                        assistant_message(error_message),
                        user_message("Either adjust your response based on the error, or continue your reasoning without using this tool."),  # noqa: E501
                    ])

        if iteration >= self.max_iterations and (not reasoning_step or reasoning_step.next_action != ReasoningAction.FINISHED):  # noqa: E501
            yield ErrorEvent(
                content=f"Maximum iterations ({self.max_iterations}) reached. Generating best answer with current information.",  # noqa: E501
                metadata={
                    'max_iterations': self.max_iterations,
                    'iteration': iteration,
                },
            )

        # Generate the final streaming response using the regular model (not structured output)
        # Create a new model instance without structured output for streaming
        final_model = Client.instantiate(
            client_type=self.client_type,
            model_name=self.model,
            **self.model_kwargs,
        )
        # Prepare the final prompt with all the reasoning history
        summary_messages = [
            system_message("Your job is to review the reasoning process and provide a final answer. Please provide the answer along with the appropriate explanation, if you deem it necessary."),  # noqa: E501
            assistant_message("Here is the user's original question:\n\n```\n" + last_message['content'] + "\n```\n"),  # noqa: E501
            assistant_message("Here is the reasoning history for the problem:\n\n```\n" + json.dumps(reasoning_history, indent=2) + "\n```\n"),  # noqa: E501
        ]
        final_answer = ""
        async for chunk in final_model.run_async(summary_messages):
            if isinstance(chunk, TextChunkEvent):
                final_answer += chunk.content
                yield chunk
            elif isinstance(chunk, ResponseSummary):
                # Update token usage stats
                total_input_tokens += chunk.input_tokens
                total_output_tokens += chunk.output_tokens
                total_input_cost += chunk.input_cost
                total_output_cost += chunk.output_cost

        # Calculate total duration
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # Yield the final summary
        yield ResponseSummary(
            response=final_answer,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            input_cost=total_input_cost,
            output_cost=total_output_cost,
            duration_seconds=duration,
        )
