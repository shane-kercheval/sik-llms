"""Fixtures for testing the sik_llms module."""
from dataclasses import dataclass
import os
from pathlib import Path
import pytest
from sik_llms import Tool, Parameter
from sik_llms.models_base import RegisteredClients


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "skip_ci: mark test to be skipped in CI environments")
    config.addinivalue_line("markers", "integration: mark test as an integration test that makes API calls")  # noqa: E501

def pytest_runtest_setup(item: pytest.Item):
    """
    Skip tests marked with skip_ci in CI environments when SKIP_CI_TESTS environment variable is
    set to true.
    """
    if (
        os.environ.get("SKIP_CI_TESTS") == "true"
        and any(marker.name == "skip_ci" for marker in item.iter_markers())
    ):
        pytest.skip("Skipping test in CI environment")


OPENAI_TEST_MODEL = 'gpt-4o-mini'
OPENAI_TEST_REASONING_MODEL = 'o3-mini'

ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'
ANTRHOPIC_TEST_THINKING_MODEL = 'claude-3-7-sonnet-latest'


@dataclass
class ClientConfig:
    """Configuration for instantiating a client."""

    client_type: RegisteredClients
    model_name: str


@pytest.fixture
def project_root() -> str:
    """Get the root directory of the project."""
    return Path(__file__).parent


@pytest.fixture
def test_files_path(project_root: str) -> str:
    """Get the path to the test_files directory."""
    return project_root / 'test_files'


@pytest.fixture
def mcp_fake_server_config(test_files_path: str) -> dict:
    return {
        "mcpServers": {
            "fake-server-text": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(test_files_path),
                    "mcp",
                    "run",
                    "mcp_fake_server_text.py",
                ],
            },
            "fake-server-calculator": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(test_files_path),
                    "mcp",
                    "run",
                    "mcp_fake_server_calculator.py",
                ],
            },
            "fake-server-misc": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(test_files_path),
                    "mcp",
                    "run",
                    "mcp_fake_server_misc.py",
                ],
            },
        },
    }


@pytest.fixture
def mcp_error_server_config(test_files_path: str) -> dict:
    return {
        "mcpServers": {
            "fake-server-error": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(test_files_path),
                    "mcp",
                    "run",
                    "mcp_fake_server_error.py",
                ],
            },
        },
    }


@pytest.fixture
def simple_weather_tool() -> Tool:
    """Create a simple weather tool with one required parameter."""
    return Tool(
        name="get_weather",
        description="Get the weather for a location.",
        parameters=[
            Parameter(
                name="location",
                param_type=str,
                required=True,
                description="The city and country for weather info.",
            ),
        ],
    )


@pytest.fixture
def complex_weather_tool() -> Tool:
    """Create a weather tool with multiple parameters including an enum."""
    return Tool(
        name="get_detailed_weather",
        description="Get detailed weather information for a location.",
        parameters=[
            Parameter(
                name="location",
                param_type=str,
                required=True,
                description="The city and country",
            ),
            Parameter(
                name="unit",
                param_type=str,
                required=False,
                description="Temperature unit",
                valid_values=["celsius", "fahrenheit"],
            ),
            Parameter(
                name="include_forecast",
                param_type=bool,
                required=False,
                description="Whether to include forecast data",
            ),
        ],
    )

@pytest.fixture
def restaurant_tool() -> Tool:
    """Create a restaurant search tool with multiple parameters."""
    return Tool(
        name="search_restaurants",
        description="Search for restaurants in a location.",
        parameters=[
            Parameter(
                name="location",
                param_type=str,
                required=True,
                description="The city to search in",
            ),
            Parameter(
                name="cuisine",
                param_type=str,
                required=False,
                description="Type of cuisine",
                valid_values=["italian", "chinese", "mexican", "indian"],
            ),
            Parameter(
                name="price_range",
                param_type=str,
                required=False,
                description="Price range",
                valid_values=["$", "$$", "$$$", "$$$$"],
            ),
            Parameter(
                name="open_now",
                param_type=bool,
                required=False,
                description="Filter for currently open restaurants",
            ),
        ],
    )
