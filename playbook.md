# sik-llms Project Playbook

*This playbook serves as the primary guide for AI coding agents and human developers to understand how to work with, maintain, and extend the sik-llms codebase.*

## 1. Project Overview

**sik-llms** is a lightweight Python library that provides an easy, consistent interface for LLM interactions across providers and functionality. The library eliminates the need to parse JSON responses or build JSON for functions/tools, supporting both OpenAI and Anthropic APIs.

### Key Technologies
- **Python 3.11+** with modern type hints
- **Pydantic** for data validation and structured output
- **AsyncIO** for async/streaming support
- **OpenAI and Anthropic APIs** as LLM providers
- **OpenTelemetry** for observability and tracing
- **MCP (Model Context Protocol)** for tool integration
- **uv** for dependency management
- **ruff** for linting and formatting
- **pytest** for testing

### Architecture Philosophy
- **Registry Pattern**: Extensible client registration system
- **Event-Driven Streaming**: Real-time response processing with typed events
- **Factory Pattern**: Simple client creation via `create_client()`
- **Provider Abstraction**: Consistent interface across different LLM providers
- **Type Safety**: Comprehensive type hints throughout the codebase

## 2. Project Structure

```
sik-llms/
├── src/sik_llms/                    # Main source code
│   ├── __init__.py                  # Public API exports
│   ├── models_base.py              # Base classes and core abstractions
│   ├── openai.py                   # OpenAI provider implementation
│   ├── anthropic.py                # Anthropic provider implementation
│   ├── reasoning_agent.py          # ReasoningAgent with iterative thinking
│   ├── mcp_manager.py              # MCP client management
│   ├── telemetry.py                # OpenTelemetry integration
│   ├── utilities.py                # Registry and utility functions
│   └── prompts/                    # Prompt templates
│       ├── reasoning_prompt.txt    # System prompt for reasoning agent
│       └── final_answer_prompt.txt # Final answer generation prompt
├── tests/                          # Test suite
│   ├── conftest.py                 # Test fixtures and configuration
│   ├── test_*.py                   # Test modules (mirror src structure)
│   └── test_files/                 # Test data and mock MCP servers
├── examples/                       # Usage examples
│   ├── examples.ipynb              # Jupyter notebook with examples
│   ├── cli.py                      # CLI tool for interactive usage
│   ├── telemetry_demo.py           # Telemetry capabilities demonstration
│   └── mcp_fake_server_config.json # Example MCP configuration
├── .github/workflows/              # CI/CD configuration
├── pyproject.toml                  # Project metadata and dependencies
├── Makefile                        # Development commands
├── ruff.toml                       # Linting and formatting configuration
└── uv.lock                         # Locked dependencies
```

### Directory Purposes
- **`src/sik_llms/`**: Core library code following single responsibility principle
- **`tests/`**: Comprehensive test suite with fixtures for different scenarios
- **`examples/`**: Real-world usage patterns and interactive tools
- **`.github/workflows/`**: CI/CD pipeline for automated testing and quality checks

### File Naming Conventions
- **Test files**: `test_<module_name>.py` mirroring source structure
- **Provider modules**: Named after the service (e.g., `openai.py`, `anthropic.py`)
- **Base classes**: Use `_base` suffix (e.g., `models_base.py`)
- **Configuration files**: Descriptive names (e.g., `mcp_fake_server_config.json`)

## 3. Getting Started

### Prerequisites
- **Python 3.11+** (required for modern type hints)
- **uv** (preferred) or pip for package management
- **API Keys**: Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` environment variables

### Environment Setup
```bash
# Clone and navigate to repository
cd ~/repos/sik-llms

# Install dependencies using uv (recommended)
uv sync

# Or install in development mode with pip
pip install -e ".[dev]"

# Verify installation
uv run python -c "from sik_llms import create_client; print('✓ Installation successful')"
```

## 4. Development Workflow

### Running the Project Locally
```bash
# Install dependencies
make build-env

# Run all quality checks and tests
make tests

# Run only fast tests (excludes integration tests)
make quicktests

# Start interactive chat with MCP tools
make chat

# List available MCP tools
make tools
```

### Development Commands
```bash
# Linting only
make linting

# Unit tests only
make unittests

# Package building
make package-build

# Full package build and publish
make package

# Telemetry development commands  
make telemetry-demo
make jaeger-start
make jaeger-stop
```

## 5. Observability and Telemetry

**sik-llms** includes comprehensive OpenTelemetry integration for observing LLM operations in production and development. Understanding the telemetry system is crucial for AI coding agents working on this codebase.

### Telemetry Architecture

The library follows OpenTelemetry standards and supports two usage patterns:

#### Zero-Config Mode (Default)
- **When**: No existing OpenTelemetry setup detected
- **Behavior**: Automatically configures OpenTelemetry with sensible defaults
- **Detection**: Uses `isinstance(provider, NoOpTracerProvider)` check
- **Configuration**: Via environment variables (`OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT`)

#### Manual Setup Mode  
- **When**: User has pre-configured OpenTelemetry
- **Behavior**: Respects existing configuration, no interference
- **Detection**: Provider is not `NoOpTracerProvider` or `NoOpMeterProvider`
- **Integration**: Uses existing tracer/meter providers

### Key Files and Components

#### `src/sik_llms/telemetry.py`
Core telemetry module containing:
- `get_tracer()`: Returns OpenTelemetry tracer with provider detection
- `get_meter()`: Returns OpenTelemetry meter with provider detection  
- `is_telemetry_enabled()`: Checks `OTEL_SDK_DISABLED` environment variable
- `safe_span()`: Context manager for safe span creation
- `create_span_link()`: Links spans for evaluation correlation
- `_get_package_version()`: Dynamic version detection

#### Provider Detection Logic
```python
# Standard approach - no custom markers
current_provider = trace.get_tracer_provider()
if not isinstance(current_provider, NoOpTracerProvider):
    # User has configured - respect their setup
    return trace.get_tracer("sik-llms") 
else:
    # Auto-configure for zero-config experience
    # ... setup TracerProvider, exporters, etc.
```

#### Integration Points
- **`models_base.py`**: Base client classes call `get_tracer()` and `get_meter()`
- **All provider clients**: Instrument LLM calls with spans and metrics
- **ReasoningAgent**: Tracks reasoning iterations and tool usage
- **TokenSummary**: Emits usage and cost metrics

### Telemetry Data Generated

#### Distributed Tracing
- **Span Names**: `llm.request`, `llm.reasoning.iteration`, `llm.tool.call`
- **Attributes**: Model name, provider, operation type, token counts, costs
- **Links**: Connect evaluation results to original generations

#### Metrics  
- **Counters**: Input/output tokens, cache hits, costs by model
- **Histograms**: Request duration, reasoning iterations
- **Labels**: Model, provider, operation type

#### Events
- **Streaming**: Text chunks, tool predictions, thinking events
- **Errors**: Failed requests with context and retry information

### Environment Variables

#### Required
- `OTEL_SDK_DISABLED`: Set to `false` to enable telemetry

#### Zero-Config Mode (Optional)
- `OTEL_SERVICE_NAME`: Service identification (default: "sik-llms")
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint (default: "http://localhost:4318")
- `OTEL_EXPORTER_OTLP_HEADERS`: Authentication headers

#### Manual Mode
- User configures OpenTelemetry in their application code
- sik-llms automatically detects and uses existing setup

### Development Guidelines

#### For AI Coding Agents

1. **Provider Detection**: Always use standard OpenTelemetry provider detection, never custom markers
2. **Graceful Degradation**: All telemetry code must work when OpenTelemetry is disabled or missing
3. **User Respect**: Never override existing user OpenTelemetry configuration
4. **Standard Patterns**: Follow OpenTelemetry ecosystem conventions

### Demo and Examples

#### `examples/telemetry_demo.py`
Comprehensive demonstration showing:
- Both setup patterns explanation
- Provider detection status
- All telemetry features (tracing, metrics, span linking)
- Multi-provider support (OpenAI + Anthropic)
- ReasoningAgent iteration tracking

#### Make Commands
```bash
make telemetry-demo    # Run full telemetry demonstration
make jaeger-start      # Start local Jaeger instance  
make jaeger-stop       # Stop local Jaeger instance
```

### Troubleshooting Common Issues

#### Provider Override Warnings
- **Symptom**: "Overriding of current TracerProvider is not allowed" warnings
- **Cause**: OpenTelemetry global state management
- **Solution**: Use provider detection to avoid setting when already configured

#### Missing Traces
- **Check**: `OTEL_SDK_DISABLED=false` is set
- **Check**: Endpoint is reachable (default: http://localhost:4318)
- **Check**: Jaeger is running for local development

#### Import Errors
- **Install**: `pip install sik-llms[telemetry]` for OpenTelemetry dependencies
- **Graceful**: Code should handle ImportError when OpenTelemetry not installed
