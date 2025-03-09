"""Fixtures for testing the sik_llms module."""
import os
import pytest
from sik_llms import Tool, Parameter

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
                type="string",
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
                type="string",
                required=True,
                description="The city and country",
            ),
            Parameter(
                name="unit",
                type="string",
                required=False,
                description="Temperature unit",
                enum=["celsius", "fahrenheit"],
            ),
            Parameter(
                name="include_forecast",
                type="boolean",
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
                type="string",
                required=True,
                description="The city to search in",
            ),
            Parameter(
                name="cuisine",
                type="string",
                required=False,
                description="Type of cuisine",
                enum=["italian", "chinese", "mexican", "indian"],
            ),
            Parameter(
                name="price_range",
                type="string",
                required=False,
                description="Price range",
                enum=["$", "$$", "$$$", "$$$$"],
            ),
            Parameter(
                name="open_now",
                type="boolean",
                required=False,
                description="Filter for currently open restaurants",
            ),
        ],
    )
