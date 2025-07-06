# sik-llms

- Easy llm interface; eliminates parsing json responses or building up json for functions/tools
    - OpenAI and any OpenAI-compatible API
    - Anthropic
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
    model_name='gpt-4o-mini',  # or e.g. 'claude-3-7-sonnet-latest'
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

## Observability with OpenTelemetry

sik-llms supports OpenTelemetry for comprehensive observability of your LLM operations.

### Quick Start

1. Install with telemetry support:
   ```bash
   pip install sik-llms[telemetry]
   ```

2. Start Jaeger for local development:
   ```bash
   docker run -d --name jaeger \
     -p 16686:16686 \
     -p 4318:4318 \
     jaegertracing/all-in-one:latest
   ```

3. Enable telemetry:
   ```bash
   export OTEL_SDK_DISABLED=false
   export OTEL_SERVICE_NAME="my-llm-app"
   ```

4. Use sik-llms normally - traces will appear at http://localhost:16686

### Configuration

Telemetry is configured via environment variables:

```bash
# Enable/disable telemetry (default: disabled)
export OTEL_SDK_DISABLED=false

# Service identification
export OTEL_SERVICE_NAME="my-application"

# OTLP endpoint (default: http://localhost:4318)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Authentication headers (if needed)
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer token"
```

### What You Get

- **Distributed tracing**: Full request flows across LLM calls
- **Performance metrics**: Token usage, latency, cost tracking
- **Error tracking**: Failed requests and their context
- **Reasoning insights**: ReasoningAgent iteration and tool usage

### Production Setup

For production, point to your observability platform:

```bash
# Honeycomb
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-api-key"

# Datadog
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.datadoghq.com"
export OTEL_EXPORTER_OTLP_HEADERS="dd-api-key=your-api-key"
```

### Development with Telemetry

For local development and testing:

```bash
# Start Jaeger for telemetry
make jaeger-start

# Run telemetry demo
make telemetry-demo

# Run tests with telemetry
make quicktests-with-telemetry    # Fast unit tests
make fulltests-with-telemetry     # Includes integration tests

# Stop Jaeger
make jaeger-stop
```

### Span Links for Evaluation

When using sik-llms with evaluation frameworks, you can link evaluation results back to original LLM generations:

```python
from sik_llms import create_span_link

# In your evaluation code
link = create_span_link(
    trace_id=response.trace_id,
    span_id=response.span_id, 
    attributes={"link.type": "evaluation_of_generation"}
)
```
