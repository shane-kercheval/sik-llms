# OpenTelemetry 101: Complete Guide for sik-llms

> **Note**: This guide was AI-generated and is intended to be a comprehensive introduction to using OpenTelemetry with sik-llms. It covers everything from basic concepts to advanced usage, with practical examples and setup instructions.

## 🎯 **What You'll Learn**

This guide will take you from "What is observability?" to "I can set up production OpenTelemetry monitoring for my LLM applications." We'll use sik-llms as our practical example throughout.

## 📚 **Table of Contents**

1. [Why Observability Matters for LLMs](#why-observability-matters-for-llms)
2. [OpenTelemetry Fundamentals](#opentelemetry-fundamentals)
3. [How sik-llms Implements OpenTelemetry](#how-sik-llms-implements-opentelemetry)
4. [Setup Guide: Jaeger (Local Development)](#setup-guide-jaeger-local-development)
5. [Setup Options: Zero-Config vs Manual](#setup-options-zero-config-vs-manual)
6. [Production Backends](#production-backends)
7. [Understanding Your Data](#understanding-your-data)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

---

## 🚨 **Why Observability Matters for LLMs**

### **The LLM Monitoring Challenge**

Imagine you're building a chatbot and users complain it's "slow and expensive." Without observability, debugging is like being blindfolded:

```python
# What users see:
response = client([{"role": "user", "content": "Help me plan a trip"}])
# "This took 8 seconds and I have no idea why!"
```

**Questions you can't answer:**
- Which LLM call took the longest? 
- How many tokens did each step use?
- Which part cost the most money?
- Did any calls fail silently?
- Is the ReasoningAgent making too many iterations?

### **With OpenTelemetry, You Get X-Ray Vision**

```
🔍 Trip Planning Request (8.2s total, $0.23 cost)
├── 🧠 ReasoningAgent.stream (6.8s)
│   ├── 💭 Thinking iteration 1 (1.2s) - "Analyzing user request"
│   ├── 🔧 Tool: search_flights (2.1s) - 150 input tokens
│   ├── 💭 Thinking iteration 2 (0.8s) - "Comparing options"  
│   ├── 🔧 Tool: search_hotels (1.9s) - 200 input tokens
│   └── 📝 Final answer (0.8s) - 300 output tokens
├── 🌐 External API calls (1.2s)
└── 📊 Response formatting (0.2s)
```

**Now you can see:**
- ✅ The flight search tool is the bottleneck (2.1s)
- ✅ Token usage per step (helps optimize prompts)
- ✅ Cost breakdown by operation
- ✅ Whether reasoning iterations are efficient

---

## 🏗️ **OpenTelemetry Fundamentals**

### **The Three Pillars of Observability**

Think of OpenTelemetry like a medical monitoring system for your code:

#### **1. Traces = The Story**
**What**: The journey of a single request through your system
**Analogy**: Like a GPS route showing every turn you took
**LLM Example**: "User asked question → Reasoning started → Called GPT-4 → Got response → Formatted output"

```python
# This creates a trace:
response = client([{"role": "user", "content": "What's 2+2?"}])

# Behind the scenes, sik-llms creates:
# Trace: "user_question_processing"
#   Span: "llm.request" (2.3s)
#     Span: "llm.openai.stream" (2.1s) 
#       Span: "http.request" (2.0s)
```

#### **2. Metrics = The Vital Signs**
**What**: Numbers that tell you system health
**Analogy**: Like heart rate, blood pressure, temperature
**LLM Example**: Token counts, costs, request rates, error rates

```python
# These automatically become metrics:
llm_tokens_input_total{model="gpt-4o-mini"} = 1,250
llm_tokens_output_total{model="gpt-4o-mini"} = 480  
llm_cost_total_usd{model="gpt-4o-mini"} = 0.0023
llm_request_duration_seconds{model="gpt-4o-mini"} = 2.3
```

#### **3. Logs = The Details** 
**What**: Detailed messages about what happened
**Analogy**: Like a diary of events
**LLM Example**: "Started reasoning", "Tool call failed", "Response parsed"

*Note: sik-llms focuses on traces and metrics. Logs are usually handled separately.*

### **Key OpenTelemetry Concepts**

#### **Spans: The Building Blocks**
```python
# A span represents one operation
with tracer.start_as_current_span("llm.request") as span:
    span.set_attribute("llm.model", "gpt-4o-mini")
    span.set_attribute("llm.tokens.input", 150)
    # ... do LLM call ...
    span.set_attribute("llm.tokens.output", 75)
```

**Span Attributes**: Key-value pairs that describe what happened
- `llm.model` = "gpt-4o-mini"
- `llm.tokens.input` = 150
- `llm.provider` = "openai"

#### **Trace Context: Connecting the Dots**
```python
# Parent span
with tracer.start_as_current_span("user.question") as parent:
    # Child span (automatically linked)
    with tracer.start_as_current_span("llm.reasoning") as child:
        # Grandchild span 
        with tracer.start_as_current_span("llm.gpt4.call"):
            pass
```

**Result**: A tree structure showing the relationships between operations.

#### **Providers: The Factories**
- **TracerProvider**: Creates tracers
- **MeterProvider**: Creates meters  
- **You usually don't touch these directly** - sik-llms handles it

#### **Exporters: The Delivery System**
- **OTLP Exporter**: Sends data using OpenTelemetry Protocol
- **Console Exporter**: Prints to terminal (for debugging)
- **Jaeger Exporter**: Sends directly to Jaeger
- **Prometheus Exporter**: Exposes metrics for Prometheus

---

## 🔧 **How sik-llms Implements OpenTelemetry**

### **Automatic Instrumentation**

sik-llms automatically creates traces for all LLM operations. Here's what happens under the hood:

```python
# Your code:
client = create_client("gpt-4o-mini")
response = client([{"role": "user", "content": "Hello"}])

# What sik-llms creates automatically:
# 
# Trace: user_request_abc123
# └── Span: llm.request (2.1s)
#     ├── Attribute: llm.model = "gpt-4o-mini"  
#     ├── Attribute: llm.provider = "openai"
#     ├── Attribute: llm.tokens.input = 8
#     ├── Attribute: llm.tokens.output = 12
#     ├── Attribute: llm.cost.total = 0.0001
#     └── Child Span: llm.openai.stream (2.0s)
#         ├── Attribute: llm.streaming = true
#         └── Attribute: llm.request.temperature = 0.7
```

### **ReasoningAgent Deep Instrumentation**

For complex reasoning, sik-llms creates detailed trace trees:

```python
# Your code:
agent = ReasoningAgent(model_name="gpt-4o-mini")
response = agent([{"role": "user", "content": "Solve: 15 * 23 + 7"}])

# What sik-llms creates:
#
# Trace: reasoning_request_def456  
# └── Span: llm.reasoning_agent.stream (8.2s)
#     ├── Attribute: llm.reasoning.effort = "medium"
#     ├── Attribute: llm.reasoning.iterations.total = 3
#     ├── Attribute: llm.reasoning.tool_calls.total = 1
#     ├── Child Span: reasoning.thinking.iteration_1 (2.1s)
#     │   └── Attribute: reasoning.thinking.content_length = 245
#     ├── Child Span: reasoning.tool.calculator (1.8s)
#     │   ├── Attribute: tool.name = "calculator"
#     │   └── Attribute: tool.arguments_count = 2
#     ├── Child Span: reasoning.thinking.iteration_2 (1.2s)
#     └── Child Span: reasoning.thinking.iteration_3 (3.1s)
```

### **Metrics Collection**

Every LLM call automatically generates metrics:

```python
# Counters (cumulative)
llm_tokens_input_total{llm_model="gpt-4o-mini", llm_provider="openai"} += 150
llm_tokens_output_total{llm_model="gpt-4o-mini", llm_provider="openai"} += 75
llm_cost_total_usd{llm_model="gpt-4o-mini", llm_provider="openai"} += 0.0023

# Histograms (distribution)  
llm_request_duration_seconds{llm_model="gpt-4o-mini"} = 2.3

# Reasoning-specific metrics
llm_reasoning_iterations_total{reasoning_effort="medium"} += 3
llm_reasoning_tool_calls_total{reasoning_effort="medium"} += 1
```

---

## 🐳 **Setup Guide: Jaeger (Local Development)**

Jaeger is the most popular open-source distributed tracing system. It's perfect for local development and understanding your traces.

### **Step 1: Start Jaeger**

**Option A: Docker (Recommended)**
```bash
# Start Jaeger all-in-one (includes UI, collector, storage)
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Verify it's running
curl http://localhost:16686/api/services
```

**Option B: Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4318:4318"    # OTLP HTTP receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

```bash
docker-compose up -d jaeger
```

**What each port does:**
- `16686`: Jaeger Web UI (where you view traces)
- `4318`: OTLP HTTP endpoint (where sik-llms sends data)

### **Step 2: Enable sik-llms Telemetry**

```bash
# Enable telemetry
export OTEL_SDK_DISABLED=false

# Optional: customize service name
export OTEL_SERVICE_NAME="my-chatbot"

# Optional: verify endpoint (this is the default)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

### **Step 3: Run Your Code**

```python
# example_with_telemetry.py
from sik_llms import create_client

# This will automatically create traces!
client = create_client("gpt-4o-mini") 
response = client([{"role": "user", "content": "What's the capital of France?"}])
print(response.response)
```

```bash
python example_with_telemetry.py
```

### **Step 4: View Your Traces**

1. **Open Jaeger UI**: http://localhost:16686
2. **Select Service**: Choose "my-chatbot" (or "sik-llms" if you didn't set a custom name)
3. **Click "Find Traces"**
4. **Explore**: Click on a trace to see the detailed span tree

**What you'll see:**
- **Timeline**: Visual representation of span durations
- **Span Details**: Attributes like model, tokens, cost
- **Logs**: Any log events attached to spans
- **Process**: Service information

### **Step 5: Understanding the Jaeger UI**

#### **Search Page**
- **Service**: Filter by service name (e.g., "my-chatbot")
- **Operation**: Filter by span names (e.g., "llm.request")
- **Tags**: Filter by attributes (e.g., `llm.model=gpt-4o-mini`)
- **Time Range**: Choose when traces occurred

#### **Trace Detail View**
```
🔍 Trace: user_request_12345 (Total: 2.3s)
├── 📊 llm.request (2.3s) 
│   ├── 🏷️ llm.model: gpt-4o-mini
│   ├── 🏷️ llm.tokens.input: 15
│   ├── 🏷️ llm.tokens.output: 8  
│   └── 📊 llm.openai.stream (2.1s)
│       ├── 🏷️ llm.streaming: true
│       └── 🏷️ llm.request.temperature: 0.7
```

#### **Useful Queries**
```bash
# Find expensive requests
llm.cost.total > 0.01

# Find slow requests  
duration > 5s

# Find reasoning operations
operation="llm.reasoning_agent.stream"

# Find failed requests
error=true
```

---

## ⚙️ **Setup Options: Zero-Config vs Manual**

### **🚀 Zero-Config Setup (Recommended for Getting Started)**

**Perfect for:**
- ✅ Prototypes and experiments
- ✅ Local development  
- ✅ Getting started quickly
- ✅ Simple applications

**How it works:**
1. sik-llms detects no existing OpenTelemetry configuration
2. Automatically sets up TracerProvider and MeterProvider
3. Configures OTLP exporter with sensible defaults
4. "Just works" with minimal setup

**Setup:**
```bash
# 1. Install telemetry support
pip install sik-llms[telemetry]

# 2. Start Jaeger
docker run -d -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one

# 3. Enable telemetry
export OTEL_SDK_DISABLED=false
export OTEL_SERVICE_NAME="my-app"

# 4. Use sik-llms normally - telemetry just works!
```

**Configuration via environment variables:**
```bash
# Service identification
export OTEL_SERVICE_NAME="my-llm-service"

# Where to send traces (default: http://localhost:4318)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Authentication (if needed)
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer your-token"
```

### **🏗️ Manual Setup (Recommended for Production)**

**Perfect for:**
- ✅ Production applications
- ✅ Applications with existing observability
- ✅ Custom OpenTelemetry configuration
- ✅ Multiple services sharing telemetry config

**How it works:**
1. You configure OpenTelemetry exactly how you want
2. sik-llms detects your configuration and respects it
3. No conflicts, no overwrites
4. Full control over sampling, exporters, resources

**Setup:**
```python
# main.py - Your application startup
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

# 1. Define your service
resource = Resource.create({
    "service.name": "my-production-app",
    "service.version": "1.2.3",
    "deployment.environment": "production",
    "team.name": "ai-platform"
})

# 2. Configure tracing
trace_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(trace_provider)

# 3. Set up trace exporter (send to your backend)
trace_exporter = OTLPSpanExporter(
    endpoint="https://api.honeycomb.io/v1/traces/your-dataset",
    headers={"x-honeycomb-team": "your-api-key"}
)
trace_processor = BatchSpanProcessor(trace_exporter)
trace_provider.add_span_processor(trace_processor)

# 4. Configure metrics
metric_exporter = OTLPMetricExporter(
    endpoint="https://api.honeycomb.io/v1/metrics/your-dataset", 
    headers={"x-honeycomb-team": "your-api-key"}
)
metric_reader = PeriodicExportingMetricReader(
    exporter=metric_exporter,
    export_interval_millis=10000  # Export every 10 seconds
)
metrics_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(metrics_provider)

# 5. Now use sik-llms - it will automatically use your configuration!
from sik_llms import create_client

client = create_client("gpt-4o-mini")
response = client([{"role": "user", "content": "Hello"}])
```

**Advanced configuration options:**
```python
# Sampling (only trace 10% of requests for performance)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
trace_provider = TracerProvider(
    resource=resource,
    sampler=TraceIdRatioBased(0.1)  # 10% sampling
)

# Multiple exporters (send to both Jaeger and Honeycomb)
jaeger_exporter = OTLPSpanExporter(endpoint="http://localhost:4318")
honeycomb_exporter = OTLPSpanExporter(
    endpoint="https://api.honeycomb.io/v1/traces/dataset",
    headers={"x-honeycomb-team": "api-key"}
)

trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace_provider.add_span_processor(BatchSpanProcessor(honeycomb_exporter))

# Custom resource attributes
resource = Resource.create({
    "service.name": "my-app",
    "service.version": os.getenv("APP_VERSION", "unknown"),
    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    "k8s.pod.name": os.getenv("HOSTNAME", "local"),
    "git.commit.sha": os.getenv("GIT_SHA", "unknown"),
})
```

### **How sik-llms Detects Your Setup**

```python
# This is what happens inside sik-llms:
from opentelemetry import trace
from opentelemetry.trace import NoOpTracerProvider

current_provider = trace.get_tracer_provider()

if isinstance(current_provider, NoOpTracerProvider):
    # No user configuration detected
    print("Using zero-config setup")
    # sik-llms sets up automatic configuration
else:
    # User has configured OpenTelemetry
    print("Using manual setup - respecting user configuration")
    # sik-llms just gets a tracer and uses existing setup
```

---

## 🌐 **Production Backends**

While Jaeger is perfect for local development, production applications typically use managed observability platforms.

### **Cloud-Native Options**

#### **Honeycomb** (Recommended for LLM applications)
```bash
# Zero-config setup
export OTEL_SDK_DISABLED=false
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io/v1/traces/your-dataset"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-api-key"
```

**Why Honeycomb for LLMs:**
- ✅ Excellent high-cardinality data support (perfect for LLM attributes)
- ✅ Great query interface for exploring traces
- ✅ Built-in cost analysis features
- ✅ Good performance with high-volume trace data

#### **Datadog**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://http-intake.logs.datadoghq.com/v1/input/your-api-key"
export OTEL_EXPORTER_OTLP_HEADERS="DD-API-KEY=your-api-key"
```

#### **New Relic**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.nr-data.net:4318/v1/traces"
export OTEL_EXPORTER_OTLP_HEADERS="api-key=your-license-key"
```

#### **AWS X-Ray** (via ADOT)
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
# Requires AWS Distro for OpenTelemetry Collector
```

#### **Google Cloud Trace**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloudtrace.googleapis.com/v1/projects/your-project/traces"
# Requires Google Cloud authentication
```

### **Self-Hosted Options**

#### **Jaeger Production Setup**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
  
  jaeger-collector:
    image: jaegertracing/jaeger-collector:latest
    environment:
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
    ports:
      - "4318:4318"  # OTLP HTTP
  
  jaeger-query:
    image: jaegertracing/jaeger-query:latest
    environment:
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
    ports:
      - "16686:16686"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

#### **Grafana + Tempo**
```yaml
# Tempo for traces, Prometheus for metrics
version: '3.8'
services:
  tempo:
    image: grafana/tempo:latest
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml
    ports:
      - "3200:3200"   # Tempo
      - "4318:4318"   # OTLP HTTP
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## 📊 **Understanding Your Data**

### **Key Metrics to Monitor**

#### **Performance Metrics**
```bash
# Request latency
llm_request_duration_seconds

# Token efficiency  
llm_tokens_input_total / llm_requests_total  # Average input tokens per request
llm_tokens_output_total / llm_requests_total # Average output tokens per request

# Cost tracking
llm_cost_total_usd

# Cache efficiency (if using caching)
llm_cache_read_tokens_total / (llm_cache_read_tokens_total + llm_tokens_input_total)
```

#### **ReasoningAgent Metrics**
```bash
# Reasoning efficiency
llm_reasoning_iterations_total / llm_reasoning_requests_total  # Avg iterations per request

# Tool usage
llm_reasoning_tool_calls_total

# Success rate
llm_reasoning_success_total / llm_reasoning_requests_total
```

### **Common Queries**

#### **In Jaeger**
```bash
# Find expensive operations
llm.cost.total > 0.05

# Find slow reasoning
operation="llm.reasoning_agent.stream" AND duration > 10s

# Find high token usage
llm.tokens.total > 1000

# Find specific models
llm.model="gpt-4o" OR llm.model="claude-3-opus"
```

#### **In Honeycomb**
```sql
-- Average response time by model
GROUP BY llm.model
CALCULATE AVG(duration_ms)

-- Token usage distribution  
VISUALIZE llm.tokens.total as heatmap(duration_ms)

-- Cost analysis
GROUP BY llm.model, llm.provider
CALCULATE SUM(llm.cost.total)

-- Reasoning iteration patterns
WHERE operation = "llm.reasoning_agent.stream"
GROUP BY llm.reasoning.iterations.total
CALCULATE COUNT
```

### **Setting Up Alerts**

#### **Key Alerts to Configure**
```yaml
# High latency
- alert: LLMHighLatency
  expr: llm_request_duration_seconds > 10
  description: "LLM request taking longer than 10 seconds"

# High cost
- alert: LLMHighCost  
  expr: rate(llm_cost_total_usd[5m]) > 0.10
  description: "LLM costs exceeding $0.10 per 5 minutes"

# High error rate
- alert: LLMHighErrorRate
  expr: rate(llm_requests_failed_total[5m]) / rate(llm_requests_total[5m]) > 0.05
  description: "LLM error rate above 5%"

# Reasoning inefficiency
- alert: ReasoningTooManyIterations
  expr: llm_reasoning_iterations_total > 10
  description: "ReasoningAgent using more than 10 iterations"
```

---

## 🔧 **Troubleshooting**

### **No Traces Appearing**

#### **1. Check if telemetry is enabled**
```bash
echo $OTEL_SDK_DISABLED
# Should output: false
```

#### **2. Verify endpoint connectivity**
```bash
# Test if Jaeger is reachable
curl http://localhost:4318/v1/traces -X POST \
  -H "Content-Type: application/json" \
  -d '{"resourceSpans":[]}'

# Should return HTTP 200
```

#### **3. Check sik-llms telemetry status**
```python
from sik_llms.telemetry import is_telemetry_enabled, get_tracer

print(f"Telemetry enabled: {is_telemetry_enabled()}")
tracer = get_tracer()
print(f"Tracer available: {tracer is not None}")

if tracer:
    print(f"Tracer type: {type(tracer)}")
```

#### **4. Verify provider detection**
```python
from opentelemetry import trace
from opentelemetry.trace import NoOpTracerProvider

provider = trace.get_tracer_provider()
print(f"Provider type: {type(provider)}")
print(f"Is NoOp: {isinstance(provider, NoOpTracerProvider)}")
```

### **Traces Not Showing Expected Data**

#### **1. Check span attributes**
```python
# Add this to your code to see what's being captured
import logging
logging.basicConfig(level=logging.DEBUG)

# Look for log messages like:
# "Creating span: llm.request with attributes: {...}"
```

#### **2. Verify manual setup isn't conflicting**
```python
# If you have manual OpenTelemetry setup, make sure it's compatible
from opentelemetry import trace

provider = trace.get_tracer_provider()
print(f"Current provider: {provider}")

# Check for span processors
if hasattr(provider, '_span_processors'):
    print(f"Span processors: {len(provider._span_processors)}")
    for processor in provider._span_processors:
        print(f"  Processor: {type(processor)}")
```

### **High Memory Usage**

#### **1. Check batch sizes**
```python
# If using manual setup, tune batch processor settings
from opentelemetry.sdk.trace.export import BatchSpanProcessor

processor = BatchSpanProcessor(
    exporter=your_exporter,
    max_queue_size=512,        # Default: 2048
    schedule_delay_millis=5000, # Default: 5000  
    max_export_batch_size=256   # Default: 512
)
```

#### **2. Consider sampling**
```python
# Only trace a percentage of requests
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

trace_provider = TracerProvider(
    sampler=TraceIdRatioBased(0.1)  # 10% sampling
)
```

### **Performance Impact**

#### **1. Measure overhead**
```python
import time
from sik_llms import create_client

client = create_client("gpt-4o-mini")

# Test with telemetry disabled
start = time.time()
for _ in range(100):
    # Your LLM operation
    pass
no_telemetry_time = time.time() - start

# Test with telemetry enabled  
# (set OTEL_SDK_DISABLED=false)
start = time.time()
for _ in range(100):
    # Same LLM operation
    pass
with_telemetry_time = time.time() - start

overhead = (with_telemetry_time - no_telemetry_time) / no_telemetry_time * 100
print(f"Telemetry overhead: {overhead:.1f}%")
```

#### **2. Optimize for production**
```python
# Reduce telemetry overhead
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Increase batch size, reduce frequency
processor = BatchSpanProcessor(
    exporter=your_exporter,
    max_export_batch_size=1024,    # Larger batches
    schedule_delay_millis=30000,   # Export every 30s instead of 5s
    export_timeout_millis=60000    # Longer timeout
)
```

---

## 🚀 **Advanced Usage**

### **Custom Instrumentation**

Add your own telemetry to complement sik-llms:

```python
from opentelemetry import trace
from sik_llms import create_client

tracer = trace.get_tracer("my_app")

def process_user_request(user_input: str):
    with tracer.start_as_current_span("user.request.processing") as span:
        span.set_attribute("user.input.length", len(user_input))
        span.set_attribute("user.input.language", detect_language(user_input))
        
        # Preprocessing
        with tracer.start_as_current_span("user.input.preprocessing"):
            clean_input = preprocess(user_input)
        
        # LLM call (automatically instrumented by sik-llms)
        client = create_client("gpt-4o-mini")
        with tracer.start_as_current_span("llm.generation") as llm_span:
            response = client([{"role": "user", "content": clean_input}])
            llm_span.set_attribute("llm.response.length", len(response.response))
        
        # Postprocessing
        with tracer.start_as_current_span("response.postprocessing"):
            final_response = postprocess(response.response)
        
        span.set_attribute("response.final.length", len(final_response))
        return final_response
```

### **Span Links for Evaluation**

Connect evaluation results back to original LLM generations:

```python
from sik_llms import create_client, create_span_link
from opentelemetry import trace

tracer = trace.get_tracer("evaluation")

# Original generation
client = create_client("gpt-4o-mini")
response = client([{"role": "user", "content": "Explain quantum computing"}])

# Later, during evaluation
with tracer.start_as_current_span("evaluation.quality_check") as eval_span:
    # Link back to original generation
    if hasattr(response, 'trace_id') and hasattr(response, 'span_id'):
        link = create_span_link(
            trace_id=response.trace_id,
            span_id=response.span_id,
            attributes={"link.type": "evaluation_of_generation"}
        )
        # Note: You'd need to add links when creating the span
        # This is a conceptual example
    
    eval_span.set_attribute("evaluation.score", 0.85)
    eval_span.set_attribute("evaluation.criteria", "accuracy")
```

### **Custom Metrics**

Add domain-specific metrics:

```python
from opentelemetry import metrics
from sik_llms import create_client

meter = metrics.get_meter("my_app")

# Custom counters
conversation_counter = meter.create_counter(
    name="conversations_total",
    description="Total number of conversations",
    unit="conversation"
)

user_satisfaction_histogram = meter.create_histogram(
    name="user_satisfaction_score",
    description="User satisfaction ratings",
    unit="score"
)

def handle_conversation(messages, user_id):
    client = create_client("gpt-4o-mini")
    
    # Increment conversation counter
    conversation_counter.add(1, {
        "user_tier": get_user_tier(user_id),
        "conversation_type": classify_conversation(messages)
    })
    
    response = client(messages)
    
    # Record satisfaction (from user feedback later)
    satisfaction = get_user_feedback(user_id, response.response)
    user_satisfaction_histogram.record(satisfaction, {
        "model": "gpt-4o-mini",
        "response_length": len(response.response)
    })
    
    return response
```

### **Multi-Service Tracing**

Trace requests across multiple services:

```python
# Service A: Web API
from opentelemetry import trace
from opentelemetry.propagators.textmap import TextMapPropagator

tracer = trace.get_tracer("web_api")

@app.route('/chat')
def chat_endpoint():
    with tracer.start_as_current_span("api.chat.request") as span:
        # Extract trace context from HTTP headers
        context = TextMapPropagator().extract(request.headers)
        
        # Call Service B with trace context
        response = call_llm_service(request.json, context)
        return response

# Service B: LLM Service  
def call_llm_service(data, context):
    # Continue the trace from Service A
    with trace.use_context(context):
        with tracer.start_as_current_span("llm_service.process"):
            client = create_client("gpt-4o-mini")
            return client(data['messages'])
```

### **Sampling Strategies**

Implement intelligent sampling for high-volume applications:

```python
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, Decision

class CostBasedSampler(Sampler):
    """Sample more expensive operations at higher rates."""
    
    def should_sample(self, parent_context, trace_id, name, kind, attributes, links, trace_state):
        # Always sample reasoning operations (they're complex)
        if "reasoning" in name.lower():
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        
        # Sample expensive models more frequently
        model = attributes.get("llm.model", "")
        if "gpt-4" in model or "claude-3-opus" in model:
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        
        # Sample cheap models less frequently
        if "gpt-3.5" in model or "gpt-4o-mini" in model:
            # 10% sampling
            return SamplingResult(Decision.RECORD_AND_SAMPLE if trace_id % 10 == 0 else Decision.DROP)
        
        # Default sampling
        return SamplingResult(Decision.RECORD_AND_SAMPLE if trace_id % 5 == 0 else Decision.DROP)

# Use the custom sampler
from opentelemetry.sdk.trace import TracerProvider
trace_provider = TracerProvider(sampler=CostBasedSampler())
```

---

## 🎓 **Next Steps**

### **Learning More**
- [OpenTelemetry Official Docs](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Observability Engineering Book](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)

### **Production Readiness**
1. **Choose your backend**: Start with Jaeger locally, move to managed service for production
2. **Set up sampling**: Avoid overwhelming your backend with 100% sampling
3. **Configure alerts**: Monitor costs, latency, and error rates
4. **Create dashboards**: Visualize your LLM usage patterns
5. **Train your team**: Ensure everyone knows how to use the observability tools

### **sik-llms Specific**
- Explore the telemetry demo: `python examples/telemetry_demo.py`
- Try both setup patterns to see which fits your needs
- Read the troubleshooting section when things go wrong
- Contribute feedback and feature requests to the sik-llms project

---

**Remember**: Observability is a journey, not a destination. Start simple with Jaeger locally, then gradually add more sophisticated monitoring as your application grows. The investment in observability pays off exponentially as your LLM application becomes more complex!
