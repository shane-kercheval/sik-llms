"""Fake MCP server with complex parameter types for testing json_schema preservation."""
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("fake-server-complex-types")


class ArgumentDefinition(BaseModel):
    """Definition of a prompt argument."""

    name: str = Field(description="Name of the argument")
    description: str | None = Field(default=None, description="Description of the argument")
    required: bool = Field(default=False, description="Whether the argument is required")


@mcp.tool()
async def update_with_arguments(
    content: str,
    arguments: list[ArgumentDefinition] | None = None,
) -> str:
    """
    Update content with optional argument definitions.

    Args:
        content: The content to update
        arguments: Optional list of argument definitions
    """
    arg_names = [arg.name for arg in arguments] if arguments else []
    return f"Updated content with arguments: {arg_names}"


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(description="Server hostname")
    port: int = Field(description="Server port")
    ssl: bool = Field(default=False, description="Use SSL")


@mcp.tool()
async def connect_server(config: ServerConfig) -> str:
    """
    Connect to a server.

    Args:
        config: Server configuration object
    """
    return f"Connected to {config.host}:{config.port}"


if __name__ == "__main__":
    mcp.run(transport='stdio')
