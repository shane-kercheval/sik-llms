"""Public facing functions and classes for models."""
from enum import Enum
from sik_llms.models_base import (
    Client,
    RegisteredClients,
    user_message,
    assistant_message,
    system_message,
    TextChunkEvent,
    ErrorEvent,
    InfoEvent,
    AgentEvent,
    ThinkingEvent,
    ThinkingChunkEvent,
    ToolPredictionEvent,
    ToolResultEvent,
    TextResponse,
    Parameter,
    Tool,
    ToolPrediction,
    ToolPredictionResponse,
    ToolChoice,
    StructuredOutputResponse,
    ReasoningEffort,
)
from sik_llms.openai import (
    OpenAI,
    OpenAITools,
    CHAT_MODEL_COST_PER_TOKEN as OPENAI_CHAT_MODEL_COST_PER_TOKEN,
)
from sik_llms.anthropic import (
    Anthropic,
    AnthropicTools,
    CHAT_MODEL_COST_PER_TOKEN as ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN,
)
from sik_llms.reasoning_agent import ReasoningAgent

def _get_client_type(model_name: str, client_type: str | Enum | None) -> str | Enum:
    if client_type:
        return client_type
    if model_name in OPENAI_CHAT_MODEL_COST_PER_TOKEN:
        return RegisteredClients.OPENAI
    if model_name in ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN:
        return RegisteredClients.ANTHROPIC
    raise ValueError(f"Unknown model name '{model_name}'")

def create_client(
        model_name: str,
        client_type: str | Enum | None = None,
        **client_kwargs: dict | None,
    ) -> Client:
    """
    Create a client instance based on the model name and type. This function is simply a
    convenience over calling Client.instantiate directly and ifers the model type from the model
    name if it is registered in the RegisteredModels enum.

    Args:
        model_name:
            The name of the model to create (e.g. 'gpt-4o-mini').
        client_type:
            The type of model to create. This can be a string or a value from RegisteredModels.
            If None, the model type is inferred from the model name. This only works for models
            registered in the RegisteredModels enum. Additionally, it will chose chat models over
            other types (e.g. `OpenAI` over `OpenAIFunctions`).
        **client_kwargs:
            Additional keyword arguments to pass to the client's constructor. Each client has
            "**model_kwargs" argument so if the keyword argument is not found in the client's
            consturcture it will be treated as a model keyword argument and sent to the model (e.g.
            temperature).

    Returns:
        A model instance.
    """
    client_type = _get_client_type(model_name, client_type)
    return Client.instantiate(client_type=client_type, model_name=model_name, **client_kwargs)


__all__ = [  # noqa: RUF022
    'create_client',
    'Client',
    'RegisteredClients',
    'user_message',
    'assistant_message',
    'system_message',
    'TextChunkEvent',
    'ErrorEvent',
    'InfoEvent',
    'AgentEvent',
    'ThinkingEvent',
    'ThinkingChunkEvent',
    'ToolPredictionEvent',
    'ToolResultEvent',
    'TextResponse',
    'Parameter',
    'Tool',
    'ToolPrediction',
    'ToolPredictionResponse',
    'ToolChoice',
    'StructuredOutputResponse',
    'ReasoningEffort',
    'ReasoningAgent',
    'OpenAI',
    'OpenAITools',
    'Anthropic',
    'AnthropicTools',
]
