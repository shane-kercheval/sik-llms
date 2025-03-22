"""Simple fake FastMCP server."""
from mcp.server.fastmcp import FastMCP
# Initialize FastMCP server
mcp = FastMCP("fake-server-calculator")

@mcp.tool()
async def calculator_sum(numbers: list[float]) -> float:
    """
    Calculate the sum of a list of numbers.

    Args:
        numbers: List of numbers to sum
    """
    return sum(numbers)

@mcp.tool()
def calculator(expression: str) -> float:
    """
    Calculate the expresion.

    Args:
        expression: a string with a simple arithmetic expression
    """
    try:
        # Only allow simple arithmetic for safety
        allowed_chars = set('0123456789+-*/() .')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e!s}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
