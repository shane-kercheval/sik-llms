# sik-llms

- Easy llm interface; eliminates parsing json responses or building up json for functions/tools
    - OpenAI and any OpenAI-compatible API
    - Anthropic
- Sync and Async support
- Functions/Tools
- Structured Output
    - Supports Anthropic models even though structured output is not natively supported in their models
- Reasoning

See [examples](https://github.com/shane-kercheval/sik-llms/blob/main/examples/examples.ipynb)

```
from sik_llms import create_client, user_message, ResponseChunk

model = create_client(
    model_name='gpt-4o-mini',  # or e.g. 'claude-3-7-sonnet-latest'
    temperature=0.1,
)
messages = [
    system_message("You are a helpful assistant."),
    user_message("What is the capital of France?"),
]

# sync
response = model(messages=messages)

# async streaming
responses = []
summary = None
async for response in model.run_async(messages=messages):
    if isinstance(response, ResponseChunk):
        print(response.content, end="")
        responses.append(response)
    else:
        summary = response

print(summary)
```

## Installation

`uv install sik-llms` or `pip install sik-llms`
