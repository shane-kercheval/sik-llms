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

sik-llms supports OpenTelemetry for comprehensive observability of your LLM operations. There are two ways to use telemetry:

### 🚀 Option 1: Zero-Config (Recommended for Quick Start)

Perfect for prototypes, scripts, and getting started quickly:

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

3. Enable telemetry via environment variables:

   ```bash
   export OTEL_SDK_DISABLED=false
   export OTEL_SERVICE_NAME="my-llm-app"
   ```

4. Use sik-llms normally - traces will appear at http://localhost:16686

**How it works**: sik-llms automatically configures OpenTelemetry with sensible defaults when no existing configuration is detected.

### 🏗️ Option 2: Manual Setup (Recommended for Production)

For production applications or when you need custom OpenTelemetry configuration:

1. Install dependencies:
   ```bash
   pip install sik-llms[telemetry]
   ```

2. Configure OpenTelemetry in your application:
   ```python
   # main.py - Your application startup
   from opentelemetry import trace
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

   # Set up your custom telemetry configuration
   trace.set_tracer_provider(TracerProvider())
   tracer_provider = trace.get_tracer_provider()

   otlp_exporter = OTLPSpanExporter(
       endpoint="https://your-telemetry-backend.com/v1/traces",
       headers={"authorization": "Bearer your-token"}
   )
   span_processor = BatchSpanProcessor(otlp_exporter)
   tracer_provider.add_span_processor(span_processor)
   ```

3. Enable telemetry via environment variables:
   ```bash
   export OTEL_SDK_DISABLED=false
   ```

4. Use sik-llms normally - it will respect your configuration

**How it works**: sik-llms detects your existing OpenTelemetry setup and uses it without any interference.

### 🔍 What You Get

Regardless of which option you choose:
- **Distributed tracing**: Full request flows across LLM calls
- **Performance metrics**: Token usage, latency, cost tracking  
- **Error tracking**: Failed requests and their context
- **Reasoning insights**: ReasoningAgent iteration and tool usage

### Configuration

Additional configuration via environment variables:

```bash
# Enable/disable telemetry (default: disabled)
export OTEL_SDK_DISABLED=false

# Service identification (for zero-config mode)
export OTEL_SERVICE_NAME="my-application"

# OTLP endpoint (for zero-config mode, default: http://localhost:4318)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Authentication headers (for zero-config mode)
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer token"
```

### Production Examples

For zero-config mode with production backends:

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

When using sik-llms with evaluation frameworks, you can link evaluation results back to original LLM generations using the TraceContext feature:

```python
# Generate content with automatic trace context
response = client([{"role": "user", "content": "Write a story about AI"}])

# Later, in your evaluation pipeline
if response.trace_context:
    link = response.trace_context.create_link({
        "link.type": "evaluation_of_generation",
        "evaluation.type": "quality_check"
    })
    
    # Use the link when creating evaluation spans
    with tracer.start_as_current_span("evaluation", links=[link] if link else []):
        quality_score = evaluate_quality(response.response)
```

### Additional Settings

#### OpenTelemetry Batch Span Processor Settings

```bash
export OTEL_BSP_SCHEDULE_DELAY=5000          # Export delay in ms
export OTEL_BSP_MAX_QUEUE_SIZE=2048          # Max queue size  
export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512    # Max batch size
export OTEL_BSP_EXPORT_TIMEOUT=30000         # Export timeout in ms
```

**Key Benefits:**
- ✅ No manual wrapper spans needed
- ✅ Works with both sync (`client()`) and async (`client.run_async()`) calls
- ✅ Automatic trace context extraction
- ✅ Clean separation between LLM operations and evaluation
