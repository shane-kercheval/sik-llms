"""Simple fake FastMCP server."""
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("fake-server-error")


@mcp.tool()
async def error_async(text: str) -> str:  # noqa: ARG001
    """
    Get the weather for a location.

    Args:
        text: the text to return.
    """
    raise Exception("An error occurred")


@mcp.tool()
def error_sync(text: str) -> str:  # noqa: ARG001
    """
    Get the weather for a location.

    Args:
        location: The city and country for weather info.
        text: the text to return.
    """
    raise Exception("An error occurred")


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

