"""Public facing functions and classes for models."""
from enum import Enum
from sik_llms.models_base import (
    Client,
    RegisteredClients,
    user_message,
    assistant_message,
    system_message,
    ChatChunkResponse,
    ChatResponseSummary,
    Parameter,
    Function,
    FunctionCallResult,
    FunctionCallResponse,
    ToolChoice,
    StructuredOutputResponse,
    ReasoningEffort,
    ContentType,
)
from sik_llms.openai import (
    OpenAI,
    OpenAIFunctions,
    CHAT_MODEL_COST_PER_TOKEN as OPENAI_CHAT_MODEL_COST_PER_TOKEN,
)
from sik_llms.anthropic import (
    Anthropic,
    AnthropicFunctions,
    CHAT_MODEL_COST_PER_TOKEN as ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN,
)


def create_client(
        model_name: str,
        client_type: str | Enum | None = None,
        **model_kwargs: dict | None,
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
        **model_kwargs:
            Additional keyword arguments to pass to the model's constructor (e.g. temperature)

    Returns:
        A model instance.
    """
    if client_type is None:
        if model_name in OPENAI_CHAT_MODEL_COST_PER_TOKEN:
            client_type = RegisteredClients.OPENAI
        elif model_name in ANTHROPIC_CHAT_MODEL_COST_PER_TOKEN:
            client_type = RegisteredClients.ANTHROPIC
        else:
            raise ValueError(f"Unknown model name '{model_name}'")
    return Client.instantiate(client_type=client_type, model_name=model_name, **model_kwargs)


__all__ = [  # noqa: RUF022
    'create_client',
    'Client',
    'RegisteredClients',
    'user_message',
    'assistant_message',
    'system_message',
    'ChatChunkResponse',
    'ChatResponseSummary',
    'Parameter',
    'Function',
    'FunctionCallResult',
    'FunctionCallResponse',
    'ToolChoice',
    'StructuredOutputResponse',
    'ReasoningEffort',
    'ContentType',
    'OpenAI',
    'OpenAIFunctions',
    'Anthropic',
    'AnthropicFunctions',
]
