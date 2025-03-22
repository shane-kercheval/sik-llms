"""Simple fake FastMCP server."""
from enum import Enum
from mcp.server.fastmcp import FastMCP
# Initialize FastMCP server
mcp = FastMCP("fake-server-misc")


class WeatherUnits(Enum):
    """Enum for weather units."""

    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"

@mcp.tool()
async def get_weather(location: str, units: WeatherUnits) -> str:
    """
    Get the weather for a location.

    Args:
        location: The city and country for weather info.
        units: The temperature unit to use (celsius or fahrenheit).
    """
    return f"The weather in {location} is 1000 degrees {units.value}."


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
