"""Reasoning agent module."""

from textwrap import dedent
import json
import asyncio
from typing import AsyncGenerator
from sik_llms.models_base import (
    Client, RegisteredClients, ResponseChunk, ResponseSummary, Tool, ToolChoice,
    ContentType, ToolPrediction, ToolPredictionResponse
)
from sik_llms.openai import (
    CHAT_MODEL_COST_PER_TOKEN as OPENAI_CHAT_MODEL_COST_PER_TOKEN,
)
from sik_llms.anthropic import (
    CHAT_MODEL_COST_PER_TOKEN as ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN,
)


@Client.register(RegisteredClients.REASONING_AGENT)
class ReasoningAgent(Client):
    """A reasoning agent that can iteratively think and execute tools."""

    def __init__(
            self,
            model_name: str,
            tools: list[Tool] | None = None,
            tool_choice: ToolChoice = ToolChoice.AUTO,
            max_iterations: int = 5,
            tool_executors: dict[str, callable] | None = None,
            reasoning_system_prompt: str | None = None,
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
        # Determine which client to use based on the model name
        if model_name in OPENAI_CHAT_MODEL_COST_PER_TOKEN:
            client_type = RegisteredClients.OPENAI
            tools_client_type = RegisteredClients.OPENAI_TOOLS
        elif model_name in ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN:
            client_type = RegisteredClients.ANTHROPIC
            tools_client_type = RegisteredClients.ANTHROPIC_TOOLS
        else:
            raise ValueError(f"Unknown model name '{model_name}'")

        # Initialize the thinking model
        self.model = model_name
        self.agent = Client.instantiate(
            client_type=client_type,
            model_name=model_name,
            **model_kwargs,
        )
        
        # Configure tools and tool execution
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.tool_executors = tool_executors or {}
        self.max_iterations = max_iterations
        
        # Create the tools agent if tools are provided
        if self.tools:
            self.tools_agent = Client.instantiate(
                client_type=tools_client_type,
                model_name=model_name,
                tools=self.tools,
                tool_choice=tool_choice,
                **model_kwargs,
            )
        
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
        6. When you have enough information, provide your final answer
        
        {tools_description}
        
        Always take your time to think and reason through the problem carefully.
        """).strip()

    async def run_async(
            self,
            messages: list[dict[str, object]],
        ) -> AsyncGenerator[ResponseChunk | ResponseSummary, None]:
        """
        Run the reasoning agent on the given messages.
        
        This method implements an iterative reasoning process where the agent:
        1. Thinks about the problem
        2. Decides to use a tool or provide an answer
        3. Executes the tool if selected
        4. Repeats until a final answer is reached or max iterations is hit
        """
        self.start_time = asyncio.get_event_loop().time()
        
        # Add the system message for reasoning
        system_message = {'role': 'system', 'content': self.reasoning_system_prompt}
        reasoning_messages = [system_message] + messages.copy()
        
        # Keep track of all reasoning steps and tool calls
        reasoning_history = []
        final_reasoning = []
        iteration = 0
        
        # Yield initial status to the client
        yield ResponseChunk(
            content=f"Reasoning about the problem...",
            content_type=ContentType.TEXT,
            iteration=iteration
        )
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Step 1: Get reasoning from the model
            thinking = None
            tool_call = None
            
            # Stream the thinking process
            full_thinking = ""
            async for chunk in self.agent.run_async(reasoning_messages):
                if isinstance(chunk, ResponseChunk):
                    if chunk.content:
                        full_thinking += chunk.content
                        yield ResponseChunk(
                            content=chunk.content,
                            content_type=ContentType.THINKING,
                            iteration=iteration,
                        )
                elif isinstance(chunk, ResponseSummary):
                    # Update token usage stats
                    self.total_input_tokens += chunk.input_tokens
                    self.total_output_tokens += chunk.output_tokens
                    self.total_input_cost += chunk.input_cost
                    self.total_output_cost += chunk.output_cost
                    thinking = chunk.content
            
            # Add the thinking to the reasoning history
            reasoning_history.append({
                'iteration': iteration,
                'thinking': thinking
            })
            final_reasoning.append(thinking)
            
            # Step 2: Try to extract tool calls if the model mentions tools
            should_use_tool = False
            tool_name = None
            tool_args = None
            
            if self.tools and any(tool.name.lower() in thinking.lower() for tool in self.tools):
                # Use the tools agent to formalize the tool call
                tool_messages = reasoning_messages.copy()
                tool_messages.append({'role': 'assistant', 'content': thinking})
                tool_messages.append({
                    'role': 'user', 
                    'content': "Based on your thinking, call the appropriate tool now."
                })
                
                tool_response = await self.tools_agent.run_async(tool_messages)
                
                # Update token usage
                self.total_input_tokens += tool_response.input_tokens
                self.total_output_tokens += tool_response.output_tokens
                self.total_input_cost += tool_response.input_cost
                self.total_output_cost += tool_response.output_cost
                
                if tool_response.tool_prediction:
                    should_use_tool = True
                    tool_call = tool_response.tool_prediction
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments
            
            # Step 3: Execute tool or provide final answer
            if should_use_tool and tool_name in self.tool_executors:
                # Yield tool call to client
                yield ResponseChunk(
                    content=f"Calling tool: {tool_name} with args: {json.dumps(tool_args, indent=2)}",
                    content_type=ContentType.TOOL_PREDICTION,
                    iteration=iteration
                )
                
                # Execute the tool
                try:
                    tool_result = await self._execute_tool(tool_name, tool_args)
                    
                    # Yield tool result to client
                    yield ResponseChunk(
                        content=f"Tool result: {tool_result}",
                        content_type=ContentType.TOOL_RESULT,
                        iteration=iteration
                    )
                    
                    # Add tool call and result to messages
                    reasoning_messages.append({'role': 'assistant', 'content': thinking})
                    reasoning_messages.append({
                        'role': 'user',
                        'content': f"Tool call: {tool_name}\nArgs: {json.dumps(tool_args)}\nResult: {tool_result}\n\nContinue your reasoning based on this tool result."
                    })
                    
                    reasoning_history.append({
                        'iteration': iteration,
                        'tool_call': {
                            'name': tool_name,
                            'args': tool_args,
                            'result': tool_result
                        }
                    })
                except Exception as e:
                    # Handle tool execution errors
                    error_message = f"Error executing tool {tool_name}: {str(e)}"
                    yield ResponseChunk(
                        content=error_message,
                        content_type=ContentType.ERROR,
                        iteration=iteration
                    )
                    
                    reasoning_messages.append({'role': 'assistant', 'content': thinking})
                    reasoning_messages.append({
                        'role': 'user',
                        'content': f"{error_message}\n\nContinue your reasoning without using this tool."
                    })
            else:
                # No tool call, or we've reached max iterations - generate final answer
                break
        
        # Step 4: Generate the final answer
        yield ResponseChunk(
            content="Generating final answer based on reasoning...",
            content_type=ContentType.TEXT
        )
        
        # Add request for final answer
        reasoning_messages.append({'role': 'assistant', 'content': thinking})
        reasoning_messages.append({
            'role': 'user',
            'content': "Based on your reasoning, provide your final answer to the original question."
        })
        
        # Get the final answer
        final_answer = ""
        async for chunk in self.agent.run_async(reasoning_messages):
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
        
        # Build the full response with reasoning
        full_response = "\n\n".join([
            final_answer,
            "\n\n---\n\n",
            "Reasoning process:",
            "\n\n".join(final_reasoning)
        ])
        
        # Yield the final summary
        yield ResponseSummary(
            content=full_response,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
            input_cost=self.total_input_cost,
            output_cost=self.total_output_cost,
            duration_seconds=duration
        )

    async def _execute_tool(self, tool_name: str, args: dict[str, object]) -> str:
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