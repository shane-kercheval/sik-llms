"""Fixtures for testing the sik_llms module."""
import pytest
from sik_llms import Function, Parameter


@pytest.fixture
def simple_weather_function() -> Function:
    """Create a simple weather function with one required parameter."""
    return Function(
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
def complex_weather_function() -> Function:
    """Create a weather function with multiple parameters including an enum."""
    return Function(
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
def restaurant_function() -> Function:
    """Create a restaurant search function with multiple parameters."""
    return Function(
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
