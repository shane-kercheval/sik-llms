"""Core MCP client functionality."""
from dataclasses import dataclass
import json
from contextlib import AsyncExitStack
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from pathlib import Path
from sik_llms.models_base import Parameter, Tool


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None

    def get_params(self) -> StdioServerParameters:
        """Convert config to StdioServerParameters."""
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )


@dataclass
class ToolInfo:
    """Information about a tool."""

    server_name: str
    tool: Tool


class MCPClientManager:
    """Manages connections to MCP servers."""

    def __init__(self, configs: str | Path | dict, silent: bool = True):
        """
        Initialize the client manager.

        Args:
            configs:
                str or Path to the location of the config file, or the configs dictionary itself.
            silent:
                If True, suppress all output during connection
        """
        self.configs = configs
        self.servers: dict[str, ClientSession] = {}
        self.tool_map: dict[str, ToolInfo] = {}
        self.exit_stack = AsyncExitStack()
        self.silent = silent

    async def __aenter__(self):
        await self.connect_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        await self.cleanup()

    @staticmethod
    def load_config(configs: str | Path | dict) -> list[ServerConfig]:
        """
        Load server configurations from config file (same format as Claude Desktop).

        Example config file:

        ```
        {
            "mcpServers": {
                "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/username/Desktop",
                    "/Users/username/Downloads"
                ]
                }
            }
        }
        ```
        """
        if isinstance(configs, str | Path):
            with open(configs) as f:
                configs = json.load(f)
        servers = []
        for name, server_config in configs.get('mcpServers', {}).items():
            servers.append(ServerConfig(
                name=name,
                command=server_config['command'],
                args=server_config['args'],
                env=server_config.get('env'),
            ))
        return servers

    async def connect_servers(self) -> None:
        """
        Connect to all servers in the config file.

        Example config file:

        ```
        {
            "mcpServers": {
                "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/username/Desktop",
                    "/Users/username/Downloads"
                ]
                }
            }
        }
        ```
        """
        configs = MCPClientManager.load_config(self.configs)
        for config in configs:
            try:
                await self.connect_server(config)
            except Exception as e:
                raise e

    async def connect_server(self, config: ServerConfig) -> None:
        """
        Connect to an MCP server and register its tools.

        Raises:
            ValueError: If a tool name conflicts with an existing tool from another server.
        """
        # Create a null error log if silent mode is enabled
        errlog = open(os.devnull, 'w') if self.silent else sys.stderr  # noqa: ASYNC230, SIM115
        # Pass the custom error log to stdio_client
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(config.get_params(), errlog=errlog),
        )

        # Close the error log when we're done with it
        if self.silent:
            self.exit_stack.callback(errlog.close)

        # Rest of the method remains the same...
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio_transport[0], stdio_transport[1]),
        )
        await session.initialize()
        self.servers[config.name] = session

        # Get and register tools
        response = await session.list_tools()
        for tool in response.tools:
            if tool.name in self.tool_map:
                existing = self.tool_map[tool.name]
                raise ValueError(
                    f"Tool name conflict: '{tool.name}' is provided by both "
                    f"'{existing.server_name}' and '{config.name}'",
                )
            self.tool_map[tool.name] = ToolInfo(
                server_name=config.name.strip(),
                tool=self._convert_to_tool(tool),
            )

    def _convert_to_tool(self, mcp_tool: object) -> Tool:
        """Convert an MCP tool to a custom Tool object."""
        tool_name = mcp_tool.name.strip()
        parameters = []

        # Get the properties and definitions from the input schema
        properties = mcp_tool.inputSchema.get('properties', {})
        required_props = mcp_tool.inputSchema.get('required', [])
        definitions = mcp_tool.inputSchema.get('$defs', {})

        # Map JSON schema types to Python types
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }

        for prop_name, prop_schema in properties.items():
            # Determine if this property is required
            is_required = prop_name in required_props

            # Check if this is a reference to a definition
            if '$ref' in prop_schema:
                # Extract the definition name from the reference
                # Format is typically "#/$defs/TypeName"
                ref_path = prop_schema['$ref'].split('/')
                if len(ref_path) > 1 and ref_path[1] == '$defs':
                    def_name = ref_path[-1]
                    if def_name in definitions:
                        # Use the referenced definition
                        definition = definitions[def_name]
                        # Get basic type
                        prop_type = definition.get('type', 'string')
                        param_type = type_mapping.get(prop_type, str)

                        # Extract enum values if present
                        valid_values = definition.get('enum')

                        # Get description from definition
                        description = definition.get('description')
                    else:
                        # Definition not found, use string as default
                        param_type = str
                        valid_values = None
                        description = None
                else:
                    # Unknown reference format, use string as default
                    param_type = str
                    valid_values = None
                    description = None
            else:
                # Handle regular properties (non-references)
                any_of_types = []
                if 'anyOf' in prop_schema:
                    for schema in prop_schema.get('anyOf', []):
                        schema_type = schema.get('type')
                        if schema_type in type_mapping:
                            any_of_types.append(type_mapping[schema_type])

                # Get the base type of the property
                prop_type = prop_schema.get('type')
                param_type = type_mapping.get(prop_type, str)

                # Extract valid values and description
                valid_values = prop_schema.get('enum')
                description = prop_schema.get('description')

            # Clean up description if present
            if description:
                description = description.strip()

            param = Parameter(
                name=prop_name,
                param_type=param_type,
                required=is_required,
                description=description,
                valid_values=valid_values,
                any_of=any_of_types if 'any_of_types' in locals() and any_of_types else None,
            )
            parameters.append(param)

        # Note that when calling the tool from the tool obeject, we will extract the text rather
        # than passing back the CallToolResult object; in other words calling the tool.func should
        # behave the same as calling the underlying function directly
        async def wrapper(**kwargs: object) -> str:
            result = await self.call_tool(tool_name, kwargs)
            if result.isError:
                raise Exception(result.content[0].text)
            return result.content[0].text

        tool_description = mcp_tool.description.strip() if mcp_tool.description else None
        return Tool(
            name=tool_name,
            parameters=parameters,
            description=tool_description,
            func=wrapper,
        )

    def get_tool_infos(self) -> list[dict]:
        """List all available tools across all servers."""
        tool_list = []
        for tool_name, tool_info in self.tool_map.items():
            # Create the tool dictionary
            tool_dict = {
                'name': tool_name,
                'server': tool_info.server_name,
                'tool': tool_info.tool,
            }
            tool_list.append(tool_dict)
        return tool_list

    def get_tools(self) -> list[Tool]:
        """Return all available tools (Tool objects) across all servers."""
        return [tool_info.tool for tool_info in self.tool_map.values()]

    def get_tool(self, tool_name: str) -> Tool:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to get
        """
        tool_info = self.tool_map.get(tool_name)
        if not tool_info:
            raise ValueError(f"Tool '{tool_name}' not found")
        return tool_info.tool

    async def call_tool(self, tool_name: str, args: dict[str, object]) -> CallToolResult:
        """
        Call a tool by name, automatically selecting the appropriate server.

        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool

        Raises:
            ValueError: If the tool name is not found
        """
        tool_info = self.tool_map.get(tool_name)
        if not tool_info:
            available_tools = ", ".join(f"'{t}'" for t in self.tool_map)
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {available_tools}",
            )

        return await self.servers[tool_info.server_name].call_tool(tool_name, args)

    async def cleanup(self) -> None:
        """Clean up all connections."""
        await self.exit_stack.aclose()
