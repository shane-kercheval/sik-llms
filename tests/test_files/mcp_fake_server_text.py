"""Simple fake FastMCP server."""
from mcp.server.fastmcp import FastMCP
# Initialize FastMCP server
mcp = FastMCP("fake-server-text")

@mcp.tool()
async def reverse_text(text: str) -> str:
    """
    Reverse the input text.

    Args:
        text: The text to reverse
    """
    return text[::-1]


@mcp.tool()
async def count_characters(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: The text to analyze
    """
    return len(text)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
