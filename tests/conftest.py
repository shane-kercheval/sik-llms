"""Fixtures for testing the sik_llms module."""
from dataclasses import dataclass
import os
import pytest
from sik_llms import Tool, Parameter
from sik_llms.models_base import RegisteredClients


OPENAI_TEST_MODEL = 'gpt-4o-mini'
OPENAI_TEST_REASONING_MODEL = 'o3-mini'

ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'
ANTRHOPIC_THINKING_MODEL = 'claude-3-7-sonnet-latest'


@dataclass
class ClientConfig:
    """Configuration for instantiating a client."""

    client_type: RegisteredClients
    model_name: str



@pytest.fixture
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def test_files_path(project_root: str) -> str:
    return os.path.join(project_root, 'tests', 'test_files')


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
