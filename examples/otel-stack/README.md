# OpenTelemetry Complete Observability Stack

A ready-to-use OpenTelemetry observability stack for sik-llms with **persistent data storage**. Includes Jaeger (traces), Prometheus (metrics), Grafana (dashboards), and an OTLP Collector.

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed
2. **Python environment** with sik-llms:
   
   **Option A: Using uv (recommended)**
   ```bash
   # Install dependencies (includes sik-llms[telemetry] from workspace)
   uv sync
   ```
   
   **Option B: Using pip**
   ```bash
   pip install "sik-llms[telemetry]"
   ```

3. **OpenAI API key** (for the example):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Step 1: Start the Observability Stack

```bash
# Start all services (Jaeger, Prometheus, Grafana, OTLP Collector)
make start

# Verify everything is healthy
make verify
```

**Access Points:**
- 📊 **Jaeger (Traces)**: http://localhost:16686
- 📈 **Prometheus (Metrics)**: http://localhost:9090  
- 📋 **Grafana (Dashboards)**: http://localhost:3000 (admin/admin)

### Step 2: Enable Telemetry

The `make test-telemetry` command handles this automatically, but for manual usage:

```bash
# Configure sik-llms to send telemetry data
export OTEL_SDK_DISABLED=false
export OTEL_SERVICE_NAME="my-llm-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

### Step 3: Run the Example

```bash
# Recommended: Use the Makefile command (automatically sets environment variables)
make test-telemetry

# Or run manually (after setting environment variables from Step 2):
uv run example_with_telemetry.py
```

**What the example demonstrates:**
- ✅ Basic LLM calls with automatic trace generation
- ✅ ReasoningAgent with detailed instrumentation
- ✅ TraceContext capture for span linking
- ✅ Evaluation workflows with trace connections
- ✅ Multi-stage pipelines

### Step 3.5: Verify OTLP Collector Pipeline (Troubleshooting)

If you suspect issues with traces not appearing in Jaeger, test the collector pipeline:

```bash
# Test that the OTLP Collector properly forwards traces to Jaeger
make verify-collector
```

This sends a test trace through the entire pipeline and verifies it reaches Jaeger.

### Step 4: Explore Your Telemetry Data

#### View Traces in Jaeger
1. Open http://localhost:16686
2. Select service: `my-llm-app`
3. Click "Find Traces"
4. Explore individual traces to see:
   - Request flows and timing
   - Token usage and costs
   - Model parameters
   - Error details

**Useful Jaeger queries:**
```
llm.cost.total > 0.01           # Expensive requests
duration > 5s                   # Slow requests  
llm.model="gpt-4o-mini"         # Specific model
operation="llm.reasoning_agent.stream"  # Reasoning operations
```

#### Query Metrics in Prometheus
1. Open http://localhost:9090
2. Try these queries:
   ```promql
   llm_tokens_input_total                    # Total input tokens
   rate(llm_cost_total_usd[5m])             # Cost per minute
   llm_request_duration_seconds             # Request latency
   sum by (llm_model) (llm_tokens_output_total)  # Tokens by model
   ```

#### Create Dashboards in Grafana
1. Open http://localhost:3000 (admin/admin)
2. Add Prometheus data source: `http://prometheus:9090`
3. Create dashboards with:
   - Token usage over time
   - Cost tracking
   - Request latency histograms
   - Error rates

## 🔧 Management Commands

### Essential Commands
```bash
make help           # Show all available commands
make start          # Start the complete stack
make stop           # Stop services (keeps data)
make status         # Show service status
make logs           # View all logs
make verify         # Health check all services
```

### Testing & Development
```bash
make test-telemetry # Test sik-llms integration
make data-info      # Show persistence status
make restart        # Restart the entire stack
```

### Data Management
```bash
# Safe operations (keep data)
make clean          # Stop containers but preserve volumes

# Selective cleaning (with confirmation prompts)
make clean-traces     # Clear only Jaeger traces
make clean-metrics    # Clear only Prometheus metrics  
make clean-dashboards # Clear only Grafana dashboards

# Nuclear option (destroys everything)
make clean-all      # DESTROY all data (with confirmation)
```

## 💾 Data Persistence

**All data persists between restarts!**

- ✅ **Jaeger Traces**: Stored using Badger database in `jaeger-badger` volume
- ✅ **Prometheus Metrics**: 30-day retention in `prometheus-data` volume  
- ✅ **Grafana Dashboards**: Settings and dashboards in `grafana-storage` volume

This means you can:
- Stop and restart services without losing historical data
- Build up metrics and traces over time
- Create and save Grafana dashboards
- Analyze long-term trends

## 🐛 Troubleshooting

### No Traces Appearing?

1. **Check telemetry is enabled:**
   ```bash
   echo $OTEL_SDK_DISABLED  # Should be 'false' or empty
   ```

2. **Verify stack is running:**
   ```bash
   make verify
   ```

3. **Check sik-llms configuration:**
   ```python
   from sik_llms.telemetry import is_telemetry_enabled
   print(f"Telemetry enabled: {is_telemetry_enabled()}")
   ```

### Services Not Starting?

```bash
# Check service status
make status

# View detailed logs
make logs

# Try restarting
make restart
```

### Port Conflicts?

The stack uses these ports:
- `16686` - Jaeger UI
- `9090` - Prometheus UI
- `3000` - Grafana UI  
- `4317/4318` - OTLP Collector

Stop any conflicting services or modify the ports in `docker-compose.yaml`.

## 🎯 Example Use Cases

### Development Workflow
```bash
# Start fresh for a new feature
make clean-traces
make start

# Develop and test your LLM application
python your_app.py

# Analyze results in Jaeger and Prometheus
# Create Grafana dashboards for monitoring
```

### Performance Analysis
```bash
# Run load tests while collecting telemetry
make start
python load_test.py

# Analyze performance in Grafana
# Identify bottlenecks in Jaeger traces
```

### Cost Monitoring
```bash
# Track LLM costs over time
# Set up Grafana alerts for cost thresholds
# Query daily/weekly cost trends in Prometheus
```

## 📚 Next Steps

1. **Read the full guide**: `../otel_101.md` for comprehensive OpenTelemetry documentation
2. **Create custom dashboards** in Grafana for your specific use cases
3. **Set up alerts** in Prometheus for cost/performance thresholds
4. **Integrate with your application** using the patterns from `example_with_telemetry.py`
5. **Explore production backends** like Honeycomb, Datadog, or managed Prometheus

## 🤝 Integration with Your Application

To add telemetry to your own sik-llms application:

```python
# 1. Enable telemetry (environment variables)
import os
os.environ["OTEL_SDK_DISABLED"] = "false"
os.environ["OTEL_SERVICE_NAME"] = "your-app-name"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

# 2. Use sik-llms normally - telemetry is automatic!
from sik_llms import create_client

client = create_client("gpt-4o-mini")
response = client([{"role": "user", "content": "Hello!"}])

# 3. Access trace context for linking (optional)
if response.trace_context:
    print(f"Trace ID: {response.trace_context.trace_id}")
    # Use for evaluation workflows, span linking, etc.
```

That's it! Traces and metrics are automatically generated and sent to your observability stack.