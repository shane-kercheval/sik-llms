# sik-llms

- Easy llm interface; eliminates parsing json responses or building up json for functions/tools
    - OpenAI and any OpenAI-compatible API
- Anthropic
- Azure OpenAI (chat, tools, Responses API helper)
- Sync and Async support
- Functions/Tools
- Structured Output
    - Supports Anthropic models even though structured output is not natively supported in their models
- Reasoning mode in OpenAI and Anthropic
- ReasoningAgent that iteratively reasons and calls tool

See [examples](https://github.com/shane-kercheval/sik-llms/blob/main/examples/examples.ipynb)

## Azure OpenAI

Azure OpenAI deployments are supported via the `AzureOpenAI` and `AzureOpenAITools` clients. Configure the endpoint and credentials with environment variables or constructor kwargs:

```
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="my-gpt-4o-mini"
export AZURE_OPENAI_API_KEY="..."  # or set AZURE_OPENAI_AD_TOKEN / provider
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
```

Simple usage mirrors the OpenAI client:

```
from sik_llms import AzureOpenAI, user_message

client = AzureOpenAI(model_name="gpt-4o-mini")
response = client(messages=[user_message("Ping from Azure")])
print(response.response)
```

- Structured output and tool calling reuse the same helpers (`response_format`, `Tool`).
- Pass `use_responses_api=True` to opt into the Azure Responses API; the wrapper will return a final `TextResponse` summary.
- Entra ID flows are supported via the `azure_ad_token` or `azure_ad_token_provider` kwargs.

```
from sik_llms import create_client, user_message, ResponseChunk

model = create_client(
    model_name='gpt-4o-mini',  # or e.g. 'claude-3-7-sonnet'
    temperature=0.1,
)
messages = [
    system_message("You are a helpful assistant."),
    user_message("What is the capital of France?"),
]

# sync
response = model(messages=messages)

# async
response = await model.run_async(messages=messages)

# async streaming
responses = []
summary = None
async for response in model.stream(messages=messages):
    if isinstance(response, TextChunkEvent):
        print(response.content, end="")
        responses.append(response)
    else:
        summary = response

print(summary)
```

## Installation

`uv install sik-llms` or `pip install sik-llms`
