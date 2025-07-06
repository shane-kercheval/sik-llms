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