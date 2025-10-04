# sik-llms

- Easy llm interface; eliminates parsing json responses or building up json for functions/tools
    - OpenAI and any OpenAI-compatible API
    - Anthropic
    - Google Gemini
- Sync and Async support
- Functions/Tools
- Structured Output
    - Supports Anthropic models even though structured output is not natively supported in their models
- Reasoning mode in OpenAI and Anthropic
- ReasoningAgent that iteratively reasons and calls tool

See [examples](https://github.com/shane-kercheval/sik-llms/blob/main/examples/examples.ipynb)

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

### Authentication

Set the relevant provider API keys in your environment before running examples or tests:

- `OPENAI_API_KEY` for OpenAI-compatible providers
- `ANTHROPIC_API_KEY` for Anthropic
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for Google Gemini

#### Getting a Google Gemini API key with `gcloud`

1. Make sure the Generative Language API is enabled for your project:

   ```bash
   gcloud services enable generativelanguage.googleapis.com --project YOUR_PROJECT_ID
   ```

2. Create or reuse an API key:

   ```bash
   gcloud services api-keys create \
     --project YOUR_PROJECT_ID \
     --display-name "sik-llms-gemini"
   ```

   To list existing keys:

   ```bash
   gcloud services api-keys list --project YOUR_PROJECT_ID
   ```

3. Fetch the key string and export it for the SDK:

   ```bash
   export GOOGLE_API_KEY=$(gcloud services api-keys get-key-string KEY_ID --project YOUR_PROJECT_ID)
   ```

   Replace `KEY_ID` with the identifier returned by `list` or `create`.

#### Minimal Gemini usage

```python
from sik_llms import create_client, user_message

client = create_client(model_name="gemini-2.5-flash")
response = client(messages=[user_message("Summarize the latest updates to the Gemini family.")])
print(response.response)
```
