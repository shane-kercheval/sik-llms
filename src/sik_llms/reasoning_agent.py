"""Reasoning agent module."""
# @dataclass
# class AgentEvent:
#     """Base class for all agent events."""

#     iteration: int


# @dataclass
# class ThinkStartEvent(AgentEvent):
#     """Agent is starting to think/reason."""

#     inputs: dict[str, object]
#     timestamp: float = field(default_factory=time.time)


# @dataclass
# class ThoughtEvent(AgentEvent):
#     """Agent produced a thought and tool prediction."""

#     thought: str
#     tool_name: str | None
#     tool_args: dict[str, object] | None
#     timestamp: float = field(default_factory=time.time)


# @dataclass
# class ToolExecutionStartEvent(AgentEvent):
#     """Agent is starting to execute a tool."""

#     tool_name: str
#     tool_args: dict[str, object]
#     timestamp: float = field(default_factory=time.time)


# @dataclass
# class ToolExecutionResultEvent(AgentEvent):
#     """Tool execution result."""

#     tool_name: str
#     tool_args: dict[str, object]
#     result: object
#     timestamp: float = field(default_factory=time.time)

