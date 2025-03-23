"""Simple fake FastMCP server."""
from enum import Enum
from mcp.server.fastmcp import FastMCP
# Initialize FastMCP server
mcp = FastMCP("fake-server-error")


@mcp.tool()
async def error_async(text: str) -> str:
    """
    Get the weather for a location.

    Args:
        location: The city and country for weather info.
        units: The temperature unit to use (celsius or fahrenheit).
    """
    raise Exception("An error occurred")


@mcp.tool()
def error_sync(text: str) -> str:
    """
    Get the weather for a location.

    Args:
        location: The city and country for weather info.
        units: The temperature unit to use (celsius or fahrenheit).
    """
    raise Exception("An error occurred")


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

