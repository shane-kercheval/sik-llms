# OpenTelemetry 101: Complete Guide for sik-llms

> **Note**: This guide was AI-generated and is intended to be a comprehensive introduction to using OpenTelemetry with sik-llms. It covers everything from basic concepts to advanced usage, with practical examples and setup instructions.

## 🎯 **What You'll Learn**

This guide will take you from "What is observability?" to "I can set up production OpenTelemetry monitoring for my LLM applications." We'll use sik-llms as our practical example throughout.

## 📚 **Table of Contents**

1. [Why Observability Matters for LLMs](#why-observability-matters-for-llms)
2. [OpenTelemetry Fundamentals](#opentelemetry-fundamentals)
3. [How sik-llms Implements OpenTelemetry](#how-sik-llms-implements-opentelemetry)
4. [Complete Observability Stack Setup](#complete-stack-traces--metrics-recommended)
5. [Quick Start: Traces Only](#quick-start-jaeger-only-traces-only)
6. [Setup Options: Zero-Config vs Manual](#setup-options-zero-config-vs-manual)
7. [Backend Alternatives: Simpler Setup Options](#backend-alternatives-simpler-setup-options)
8. [Production Backends (Detailed Configuration)](#production-backends-detailed-configuration)
9. [Understanding Your Data](#understanding-your-data)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)

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

- ✅ The search tools (flight and hotel) are the bottleneck (2.1s & 1.9s)
- ✅ Token usage per step (helps optimize prompts)
- ✅ Cost breakdown by operation
- ✅ Whether reasoning iterations are efficient

---

## 🏗️ **OpenTelemetry Fundamentals**

### **The Three Pillars of Observability**

OpenTelemetry defines three pillars of observability:

#### **1. Traces = The Story (Built from Spans)**

**What**: The journey of a single request through your system
**How**: Made up of **spans** - individual operations with start/end times
**LLM Example**: "User asked question → Reasoning started → Called GPT-4 → Got response → Formatted output"

```python
# This creates a TRACE made up of multiple SPANS:
response = client([{"role": "user", "content": "What's 2+2?"}])

# Behind the scenes, sik-llms creates spans:
# Trace: "user_question_processing"
#   Span: "llm.request" (2.3s)           ← Created by a span
#     Span: "llm.openai.stream" (2.1s)   ← Created by a span
#       Span: "http.request" (2.0s)      ← Created by a span
```

#### **2. Metrics = The Vital Signs**

**What**: Numbers that tell you system health
**LLM Example**: Token counts, costs, request rates, error rates

sik-llms automatically sends metrics like token usage, costs, and request duration to your observability backend (e.g., Prometheus). You can then query and visualize these metrics to understand your LLM usage patterns, set up alerts, and create dashboards.

#### **Logs (3rd Pillar)**

`sik-llms` does not emit OpenTelemetry logs or span events. If you need logging, you'd add your own using Python's `logging` module or OpenTelemetry's logging APIs.

### **Key OpenTelemetry Concepts**

#### **Spans: The Building Blocks**

```python
# A span represents one operation
# (example of sik-llms internal span) 
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
  # (example of sik-llms internal span)
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

### **Fully Automatic Telemetry**

**The key thing to understand**: sik-llms automatically emits **both traces and metrics** for every LLM call. You don't need to instrument anything in your application code.

```python
# Your code:
client = create_client("gpt-4o-mini")
response = client([{"role": "user", "content": "Hello"}])

# What sik-llms creates automatically:
#
# 1. TRACES (sent to localhost:4318/v1/traces)
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
#
# 2. METRICS (sent to localhost:4318/v1/metrics) 
# llm_tokens_input_total{llm_model="gpt-4o-mini", llm_provider="openai"} += 8
# llm_tokens_output_total{llm_model="gpt-4o-mini", llm_provider="openai"} += 12
# llm_cost_total_usd{llm_model="gpt-4o-mini", llm_provider="openai"} += 0.0001
# llm_request_duration_seconds{llm_model="gpt-4o-mini"} = 2.1
#
# 3. TRACE CONTEXT (NEW! - automatically captured in response)
# response.trace_context.trace_id = "abc123..."
# response.trace_context.span_id = "def456..."
```

### **TraceContext**

Every response object automatically includes trace context when telemetry is enabled:

```python
response = client([{"role": "user", "content": "Hello"}])
# True if telemetry is enabled
if response.trace_context:
    print(f"Trace ID: {response.trace_context.trace_id}")
    print(f"Span ID: {response.trace_context.span_id}")
    # Create links for downstream evaluation
    link = response.trace_context.create_link({
        "link.type": "evaluation",
        "evaluation.system": "quality_checker"
    })
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
#
# PLUS: response.trace_context automatically populated
```

### **Automatic Metrics Collection**

**Important**: sik-llms automatically emits these metrics for every LLM call. You don't write any code - this happens behind the scenes:

**Token Usage:**
- `llm_tokens_input_total` - Total input tokens consumed
- `llm_tokens_output_total` - Total output tokens generated  

**Performance:**
- `llm_request_duration_seconds` - Request latency

**Cost Tracking:**
- `llm_cost_total_usd` - Total cost in USD

**Cache Metrics (when using caching):**
- `llm_cache_read_tokens_total` - Tokens read from cache
- `llm_cache_write_tokens_total` - Tokens written to cache

**ReasoningAgent Metrics:**
- `llm_reasoning_iterations_total` - Number of reasoning iterations
- `llm_reasoning_tool_calls_total` - Number of tool calls

These metrics are sent to your observability backend (e.g. Prometheus, Honeycomb, etc.) with appropriate labels like model name and provider.

---

## 🐳 **Setup Guide: Complete Observability Stack**

### **Understanding OpenTelemetry Backend Options**

OpenTelemetry data flows to **backend systems that support OTLP (OpenTelemetry Protocol)**. sik-llms is **backend-agnostic** - it just sends standardized OTLP data to any compatible system.

**In this guide, we use popular open-source examples**:
1. **Traces** → **Jaeger** (for request flows and debugging)
2. **Metrics** → **Prometheus** (for monitoring and alerting)

```
sik-llms → OTLP Collector → ┌─ Jaeger (traces)
                           └─ Prometheus (metrics) → Grafana (dashboards)
```

**Alternative backends** (that can often handle both traces AND metrics with simpler setup):
- **Honeycomb** - Handles traces + metrics in one system
- **Datadog** - Full observability platform
- **New Relic** - Application performance monitoring
- **AWS X-Ray** + CloudWatch - AWS native
- **Google Cloud Trace** + Monitoring - GCP native
- **Grafana Cloud** - Managed Grafana + Prometheus + Loki

**Why we use the two-backend example**: Jaeger + Prometheus demonstrates the full OpenTelemetry ecosystem and gives you maximum flexibility, but **other backends can simplify this significantly**.

---

## 🎯 **Quick Start: Traces Only (Jaeger Example)**

**For traces only, using Jaeger as a common example backend**:

**Why this works simply**: Jaeger all-in-one includes a built-in OTLP receiver that can directly accept trace data from sik-llms.

### **Step 1: Start Jaeger**

**Quick Jaeger Setup**
```bash
# Start Jaeger all-in-one (traces only)
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

# This will automatically create traces AND populate trace context!
client = create_client("gpt-4o-mini") 
response = client([{"role": "user", "content": "What's the capital of France?"}])

print(f"Response: {response.response}")

# Check trace context
if response.trace_context:
    print(f"🔗 Trace ID: {response.trace_context.trace_id}")
    print(f"🔗 Span ID: {response.trace_context.span_id}")
    print("✅ Ready for span linking!")
else:
    print("⚪ No trace context (telemetry may be disabled)")
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
- **Process**: Service information

*Note: sik-llms does not emit span events, so you won't see "logs" in Jaeger spans.*

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

## 📊 **Complete Stack: Traces + Metrics (Jaeger + Prometheus Example)**

**This section shows how to set up the full observability stack using popular open-source tools**. Remember, these are just examples - you can substitute other OTLP-compatible backends.

**Why this example needs multiple services**:
- **Jaeger** handles traces but not metrics
- **Prometheus** handles metrics but not traces  
- **OTLP Collector** routes the data appropriately
- **Alternative**: Single backends like Honeycomb can replace this entire stack

### **Step 1: Complete Docker Compose Setup**

```yaml
# docker-compose.yml - Complete observability stack
version: '3.8'
services:
  # Jaeger - for traces
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4318:4318"    # OTLP HTTP receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true
  
  # Prometheus - for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"    # Prometheus UI
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--web.enable-lifecycle'
  
  # OTLP Collector - routes data to backends
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otelcol-contrib/otel-collector.yml"]
    volumes:
      - ./otel-collector.yml:/etc/otelcol-contrib/otel-collector.yml
    ports:
      - "4317:4317"    # OTLP gRPC receiver  
      - "4318:4318"    # OTLP HTTP receiver
    depends_on:
      - jaeger
      - prometheus
  
  # Grafana - for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"    # Grafana UI
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### **Step 2: Create Configuration Files**

**prometheus.yml**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']  # Collector metrics endpoint
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**otel-collector.yml**
```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  # Send traces to Jaeger
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  # Send metrics to Prometheus
  prometheus:
    endpoint: "0.0.0.0:8889"
    resource_to_telemetry_conversion:
      enabled: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
    
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### **Step 3: Start the Complete Stack**

```bash
# Create the config files first (copy the YAML above)
mkdir -p otel-stack
cd otel-stack

# Create docker-compose.yml, prometheus.yml, and otel-collector.yml
# (copy the content from above)

# Start everything
docker-compose up -d

# Verify all services are running
docker-compose ps
```

### **Step 4: Access Your Observability UIs**

- **Jaeger (Traces)**: http://localhost:16686
- **Prometheus (Metrics)**: http://localhost:9090  
- **Grafana (Dashboards)**: http://localhost:3000 (admin/admin)

### **Step 5: Query Your Metrics**

Now you can actually see those `llm_*` metrics!

**In Prometheus UI (localhost:9090)**:
```promql
# Query examples:
llm_tokens_input_total
llm_request_duration_seconds
rate(llm_cost_total_usd[5m])
sum by (llm_model) (llm_tokens_output_total)
```

**In Grafana (localhost:3000)**:
1. Add Prometheus as data source: `http://prometheus:9090`
2. Create dashboards with charts showing:
   - Token usage over time
   - Cost per model
   - Request latency histograms
   - Error rates

### **Step 6: Enable sik-llms Telemetry**

```bash
# Enable telemetry to send to the collector
export OTEL_SDK_DISABLED=false
export OTEL_SERVICE_NAME="my-llm-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Run your sik-llms code
python your_llm_app.py
```

**How sik-llms handles endpoints**:
- **Traces** → `http://localhost:4318/v1/traces`
- **Metrics** → `http://localhost:4318/v1/metrics`

*sik-llms automatically appends the correct paths based on your base endpoint.*

**Now you'll see** (in our Jaeger + Prometheus example):
- ✅ **Traces** in Jaeger showing request flows
- ✅ **Metrics** in Prometheus showing token usage, costs, latency
- ✅ **Dashboards** in Grafana visualizing your LLM performance
- ✅ **TraceContext** automatically captured in response objects

**With alternative backends**: You might get all of this in a single UI (e.g., Honeycomb combines traces and metrics).

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

# 2. Start Jaeger (traces only)
docker run -d -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one

# 3. Enable telemetry
export OTEL_SDK_DISABLED=false
export OTEL_SERVICE_NAME="my-app"
# Default endpoint: http://localhost:4318 (sik-llms adds /v1/traces and /v1/metrics)

# 4. Use sik-llms normally - telemetry just works!
```

**Note about zero-config**: This setup only collects **traces** because Jaeger doesn't handle metrics. For metrics, see the complete stack below, or consider backends like Honeycomb/Datadog that handle both traces and metrics in one system.

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

# TraceContext is still automatically populated
if response.trace_context:
    print("✅ Custom OpenTelemetry setup + automatic TraceContext!")
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

---

## 🌍 **Backend Alternatives: Simpler Setup Options**

The Jaeger + Prometheus example above demonstrates the full OpenTelemetry ecosystem, but **many backends can simplify this dramatically**.

### **Single-Backend Solutions (Easier Setup)**

#### **Honeycomb** (Recommended for simplicity)
```bash
# One backend handles both traces AND metrics
export OTEL_SDK_DISABLED=false
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io/v1/traces/your-dataset"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-api-key"

# That's it! Both traces and metrics go to one place
# PLUS: TraceContext automatically works
```

#### **Datadog**
```bash
export OTEL_SDK_DISABLED=false
export OTEL_EXPORTER_OTLP_ENDPOINT="https://http-intake.logs.datadoghq.com/v1/input/your-api-key"
export OTEL_EXPORTER_OTLP_HEADERS="DD-API-KEY=your-api-key"
```

#### **New Relic**
```bash
export OTEL_SDK_DISABLED=false
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.nr-data.net:4318/v1/traces"
export OTEL_EXPORTER_OTLP_HEADERS="api-key=your-license-key"
```

### **Why Single Backends Are Simpler**

**With Jaeger + Prometheus** (our example):
- Need 4 services: Jaeger + Prometheus + Grafana + OTLP Collector
- Separate UIs for traces vs metrics
- Complex configuration files

**With Honeycomb/Datadog/etc.**:
- One service receives everything
- One UI for traces AND metrics correlation
- Simple environment variable configuration
- TraceContext still works automatically

### **When to Use Each Approach**

**Use Jaeger + Prometheus when**:
- You want full control and customization
- You prefer open-source solutions
- You're building internal observability platforms
- You want to understand the full OpenTelemetry ecosystem

**Use single backends when**:
- You want the simplest possible setup
- You prefer managed services
- You need advanced correlation between traces and metrics
- You want enterprise features (alerting, anomaly detection, etc.)

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

# In BOTH cases: TraceContext is automatically populated!
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
- ✅ Works seamlessly with sik-llms TraceContext

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

### **Where to Find Your Data**

**Traces**: View in Jaeger UI (http://localhost:16686)
**Metrics**: Query in Prometheus (http://localhost:9090) or visualize in Grafana (http://localhost:3000)
**TraceContext**: Automatically captured in response objects for span linking

### **Key Metrics to Monitor**

**Important**: These are **Prometheus queries** you run in the Prometheus UI or use in Grafana dashboards. sik-llms automatically sends these metrics to Prometheus.

#### **Performance Metrics**
```promql
# Request latency (Prometheus query)
llm_request_duration_seconds

# Token efficiency (calculated in Prometheus)
rate(llm_tokens_input_total[5m]) / rate(llm_requests_total[5m])   # Avg input tokens/request
rate(llm_tokens_output_total[5m]) / rate(llm_requests_total[5m])  # Avg output tokens/request

# Cost tracking
llm_cost_total_usd
rate(llm_cost_total_usd[1h])  # Cost per hour

# Cache efficiency (if using caching)
llm_cache_read_tokens_total / (llm_cache_read_tokens_total + llm_tokens_input_total)
```

#### **ReasoningAgent Metrics**
```promql
# Reasoning efficiency (Prometheus queries)
rate(llm_reasoning_iterations_total[5m]) / rate(llm_reasoning_requests_total[5m])  # Avg iterations per request

# Tool usage
llm_reasoning_tool_calls_total
rate(llm_reasoning_tool_calls_total[5m])  # Tool calls per minute

# Success rate
rate(llm_reasoning_success_total[5m]) / rate(llm_reasoning_requests_total[5m])
```

### **Common Queries**

#### **In Jaeger (Trace Queries)**
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

#### **In Prometheus (Metrics Queries)**
```promql
# Average response time by model
avg by (llm_model) (llm_request_duration_seconds)

# Token usage over time
rate(llm_tokens_input_total[5m])
rate(llm_tokens_output_total[5m])

# Cost analysis
sum by (llm_model, llm_provider) (rate(llm_cost_total_usd[1h]))

# Request rate
rate(llm_requests_total[5m])

# Error rate
rate(llm_requests_failed_total[5m]) / rate(llm_requests_total[5m])

# Reasoning iteration patterns
avg by (reasoning_effort) (llm_reasoning_iterations_total)
```

#### **In Grafana (Dashboard Queries)**
```promql
# Time series panels:
rate(llm_tokens_input_total{llm_model="$model"}[5m])
histogram_quantile(0.95, llm_request_duration_seconds_bucket)

# Single stat panels:
sum(rate(llm_cost_total_usd[24h])) * 24  # Daily cost
avg(llm_request_duration_seconds)        # Average latency
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

### **TraceContext Not Populated**

#### **1. Check TraceContext in response**
```python
response = client([{"role": "user", "content": "Hello"}])

print(f"Has trace_context: {hasattr(response, 'trace_context')}")
print(f"TraceContext value: {response.trace_context}")

if response.trace_context:
    print(f"Trace ID: {response.trace_context.trace_id}")
    print(f"Span ID: {response.trace_context.span_id}")
else:
    print("No trace context - check telemetry setup")
```

#### **2. Test context extraction directly**
```python
from sik_llms.telemetry import extract_current_trace_context

trace_id, span_id = extract_current_trace_context()
print(f"Direct extraction: trace_id={trace_id}, span_id={span_id}")
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

### **🆕 Simplified Span Linking with TraceContext**

Use the TraceContext feature to simplify span linking. Example below shows an evaluation workflow:

```python
from opentelemetry import trace
from sik_llms import create_client

tracer = trace.get_tracer("evaluation_system")

def evaluate_llm_generation():
    """Modern evaluation workflow using TraceContext."""
    
    # 1. Generate content (trace context automatically captured)
    client = create_client("gpt-4o-mini")
    response = client([{
        "role": "user", 
        "content": "Write a creative story about space exploration"
    }])
    
    print(f"Generated: {response.response[:100]}...")
    
    # 2. Later, run evaluation with automatic span linking
    if response.trace_context:
        # Create link back to original generation
        evaluation_link = response.trace_context.create_link({
            "link.type": "evaluation_of_generation",
            "evaluation.category": "creativity", 
            "evaluation.criteria": "originality,engagement,coherence"
        })
        
        # Run evaluation with the link
        links = [evaluation_link] if evaluation_link else []
        with tracer.start_as_current_span("content_evaluation", links=links) as eval_span:
            # Your evaluation logic
            creativity_score = assess_creativity(response.response)
            engagement_score = assess_engagement(response.response)
            coherence_score = assess_coherence(response.response)
            
            # Record evaluation results
            eval_span.set_attribute("evaluation.creativity_score", creativity_score)
            eval_span.set_attribute("evaluation.engagement_score", engagement_score) 
            eval_span.set_attribute("evaluation.coherence_score", coherence_score)
            eval_span.set_attribute("evaluation.overall_score", 
                                   (creativity_score + engagement_score + coherence_score) / 3)
            
            print(f"✅ Evaluation complete - linked to trace {response.trace_context.trace_id}")
            return {
                "creativity": creativity_score,
                "engagement": engagement_score,
                "coherence": coherence_score
            }
    else:
        print("⚠️  No trace context available - evaluation will not be linked")
        return None

# Run the evaluation
results = evaluate_llm_generation()
```

- ✅ **No manual wrapper spans** - context captured automatically
- ✅ **Works with any LLM call** - sync, async, reasoning, tools, etc.
- ✅ **Clean separation** - generation and evaluation are separate spans but linked
- ✅ **Easy to use** - just check `response.trace_context` and call `create_link()`

### **Example: Time-Separated Evaluation with TraceContext**

The real power of TraceContext comes when you generate responses at one time and evaluate them later. **Without TraceContext**, you lose the connection between evaluation results and the original LLM calls.

```python
def generate_and_save_responses(test_cases):
    """Generate responses and save them for later evaluation."""
    
    client = create_client("gpt-4o-mini")
    saved_responses = []
    
    for i, test_case in enumerate(test_cases):
        # Generate response (creates its own trace automatically)
        response = client(test_case["messages"])
        
        # Save response + trace context for later
        saved_responses.append({
            "case_id": i,
            "response": response.response,
            "expected": test_case["expected"],
            "trace_id": response.trace_context.trace_id if response.trace_context else None,
            "span_id": response.trace_context.span_id if response.trace_context else None,
        })
        
        print(f"Generated response {i}, trace: {response.trace_context.trace_id}")
    
    return saved_responses

def evaluate_saved_responses(saved_responses):
    """Hours or days later, evaluate the saved responses."""
    
    tracer = trace.get_tracer("evaluation_system")
    
    for saved in saved_responses:
        # Create link back to the original generation (from hours ago!)
        link = None
        if saved["trace_id"] and saved["span_id"]:
            link = create_span_link(
                saved["trace_id"], 
                saved["span_id"], 
                {
                    "link.type": "delayed_evaluation",
                    "case_id": saved["case_id"]
                }
            )
        
        # Run evaluation with link to original generation
        with tracer.start_as_current_span("evaluate_saved_response", links=[link] if link else []):
            score = evaluate_response(saved["response"], saved["expected"])
            
            if score < 7.0:  # Poor score - need to investigate!
                print(f"⚠️  Case {saved['case_id']} scored {score}")
                print(f"🔗 Original generation trace: {saved['trace_id']}")
                # In Jaeger: click the link to jump to the original LLM call!

# Usage scenario:
# Monday morning: Generate all responses
saved_responses = generate_and_save_responses(test_cases)

# Monday afternoon: Evaluate them with full traceability
evaluate_saved_responses(saved_responses)
```

#### **Why This Matters**

**Scenario**: You notice evaluation case #47 got a terrible score and want to debug why.

**Without TraceContext**:
- ❌ You have the evaluation result but no connection to the original LLM call
- ❌ "Which trace was case #47? Let me search through hundreds of traces..."
- ❌ Maybe grep through logs or re-run the case to debug

**With TraceContext**:
- ✅ Click the link in the evaluation span in Jaeger
- ✅ Jump directly to the original generation trace from hours ago
- ✅ See the exact prompt, model parameters, token usage, response time, cost
- ✅ Determine if it was a prompt issue, model issue, or evaluation issue

**The key insight**: TraceContext creates a "time machine" that lets you jump from any evaluation result back to the exact moment the content was generated, no matter how much time has passe

### **Advanced Multi-Stage Workflows**

```python
def multi_stage_content_pipeline():
    """Complex workflow with multiple LLM calls and automatic linking."""
    
    client = create_client("gpt-4o-mini")
    
    # Stage 1: Initial content generation
    with tracer.start_as_current_span("pipeline.content_generation") as pipeline_span:
        initial_response = client([{
            "role": "user",
            "content": "Create an outline for a blog post about AI ethics"
        }])
        
        # Stage 2: Expand the outline  
        expansion_response = client([{
            "role": "user",
            "content": f"Expand this outline into a full blog post:\n{initial_response.response}"
        }])
        
        # Stage 3: Review and edit
        final_response = client([{
            "role": "user",
            "content": f"Review and improve this blog post:\n{expansion_response.response}"
        }])
        
        # Now link all stages for analysis
        stages = [
            ("outline", initial_response),
            ("expansion", expansion_response), 
            ("final", final_response)
        ]
        
        for stage_name, response in stages:
            if response.trace_context:
                # Link each stage back to the main pipeline
                link = response.trace_context.create_link({
                    "link.type": "pipeline_stage",
                    "pipeline.stage": stage_name,
                    "pipeline.id": "content_creation_001"
                })
                
                if link:
                    # Could store these links for later analysis
                    pipeline_span.set_attribute(f"pipeline.{stage_name}.trace_id", 
                                               response.trace_context.trace_id)
        
        return final_response

# Usage
final_content = multi_stage_content_pipeline()
print(f"Final content trace: {final_content.trace_context.trace_id if final_content.trace_context else 'None'}")
```

### **Custom Instrumentation (Complementing sik-llms)**

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
        
        # LLM call (automatically instrumented by sik-llms + gets TraceContext)
        client = create_client("gpt-4o-mini")
        with tracer.start_as_current_span("llm.generation") as llm_span:
            response = client([{"role": "user", "content": clean_input}])
            llm_span.set_attribute("llm.response.length", len(response.response))
            
            # Can access trace context for linking
            if response.trace_context:
                llm_span.set_attribute("llm.trace_id", response.trace_context.trace_id)
        
        # Postprocessing
        with tracer.start_as_current_span("response.postprocessing"):
            final_response = postprocess(response.response)
        
        span.set_attribute("response.final.length", len(final_response))
        return final_response, response.trace_context
```

### **Integration with Existing Evaluation Frameworks**

```python
# Example: Integrating with a hypothetical evaluation framework
class LLMEvaluator:
    def __init__(self):
        self.tracer = trace.get_tracer("llm_evaluator")
        self.client = create_client("gpt-4o-mini")
    
    def evaluate_with_context(self, prompt: str, expected_criteria: dict):
        """Evaluate LLM response with automatic trace linking."""
        
        # Generate response
        response = self.client([{"role": "user", "content": prompt}])
        
        # Run evaluation with optional linking
        evaluation_links = []
        if response.trace_context:
            link = response.trace_context.create_link({
                "link.type": "quality_evaluation",
                "evaluation.framework": "custom_evaluator",
                "evaluation.version": "1.0"
            })
            if link:
                evaluation_links.append(link)
        
        with self.tracer.start_as_current_span("evaluation.quality", links=evaluation_links) as eval_span:
            # Run your evaluation metrics
            scores = {}
            for criterion, weight in expected_criteria.items():
                score = self._evaluate_criterion(response.response, criterion)
                scores[criterion] = score
                eval_span.set_attribute(f"evaluation.{criterion}.score", score)
                eval_span.set_attribute(f"evaluation.{criterion}.weight", weight)
            
            # Overall score
            overall = sum(score * expected_criteria[criterion] for criterion, score in scores.items())
            eval_span.set_attribute("evaluation.overall_score", overall)
            
            return {
                "response": response.response,
                "scores": scores,
                "overall": overall,
                "trace_context": response.trace_context,
                "linked": bool(evaluation_links)
            }

# Usage
evaluator = LLMEvaluator()
result = evaluator.evaluate_with_context(
    prompt="Explain quantum computing simply",
    expected_criteria={"clarity": 0.4, "accuracy": 0.4, "completeness": 0.2}
)

print(f"Evaluation score: {result['overall']:.2f}")
print(f"Linked to trace: {result['linked']}")
```

### **Custom Metrics (Beyond sik-llms Automatic Metrics)**

**Important**: sik-llms already emits all core LLM metrics automatically (tokens, costs, latency, etc.). This section is for adding **additional domain-specific metrics** to your application.

```python
from opentelemetry import metrics
from sik_llms import create_client

# Get the same meter that sik-llms uses (or create your own)
meter = metrics.get_meter("my_app")

# Custom business metrics (NOT provided by sik-llms)
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

trace_context_usage_counter = meter.create_counter(
    name="trace_context_usage_total",
    description="Number of times TraceContext was used for linking",
    unit="usage"
)

def handle_conversation(messages, user_id):
    client = create_client("gpt-4o-mini")
    
    # Your custom business metrics
    conversation_counter.add(1, {
        "user_tier": get_user_tier(user_id),
        "conversation_type": classify_conversation(messages)
    })
    
    # sik-llms automatically emits: llm_tokens_*, llm_cost_*, llm_request_duration_*, etc.
    response = client(messages)
    
    # Track TraceContext usage
    if response.trace_context:
        trace_context_usage_counter.add(1, {
            "provider": "openai" if "gpt" in client.model_name else "other",
            "has_evaluation": "true"  # if this will be evaluated later
        })
    
    # More custom metrics
    satisfaction = get_user_feedback(user_id, response.response)
    user_satisfaction_histogram.record(satisfaction, {
        "model": client.model_name,
        "response_length": len(response.response),
        "had_trace_context": str(bool(response.trace_context))
    })
    
    return response
```

### **Multi-Service Tracing with TraceContext**

Trace requests across multiple services while preserving TraceContext:

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
        
        # Response includes TraceContext automatically
        return {
            "response": response.response,
            "trace_info": {
                "trace_id": response.trace_context.trace_id if response.trace_context else None,
                "linked": bool(response.trace_context)
            }
        }

# Service B: LLM Service  
def call_llm_service(data, context):
    # Continue the trace from Service A
    with trace.use_context(context):
        with tracer.start_as_current_span("llm_service.process"):
            client = create_client("gpt-4o-mini")
            response = client(data['messages'])
            
            # TraceContext is automatically populated and inherits the distributed trace
            return response
```
