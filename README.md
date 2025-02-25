# sik-llms

Easy llm interface. Sync and Async support.

```
from sik_llms import create_client, user_message, ChatChunkResponse

model = create_client(
    model_name='gpt-4o-mini',  # or e.g. 'claude-3-7-sonnet-latest'
    temperature=0.1,
)
message = user_message("What is the capital of France?")

# sync
response = model(messages=[message])

# async streaming
responses = []
summary = None
async for response in model.run_async(messages=[message]):
    if isinstance(response, ChatChunkResponse):
        print(response.content, end="")
        responses.append(response)
    else:
        summary = response

print(summary)
```

## Installation

`uv install sik-llms` or `pip install sik-llms`
