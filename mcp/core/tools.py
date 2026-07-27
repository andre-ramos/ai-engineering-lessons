import json
from typing import Any, Optional
from mcp.types import CallToolResult, TextContent
from openai.types.responses import Response

from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(
        cls, clients: dict[str, MCPClient]
    ) -> list[dict[str, Any]]:
        """Gets all tools from the provided clients."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            for tool_model in tool_models:
                tool = {
                    "type": "function",
                    "name": tool_model.name,
                    "parameters": tool_model.inputSchema,
                }
                if tool_model.description:
                    tool["description"] = tool_model.description
                tools.append(tool)
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_part(
        cls,
        call_id: str,
        text: str,
    ) -> dict[str, str]:
        """Builds a tool result part dictionary."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": text,
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], response: Response
    ) -> list[dict[str, str]]:
        """Executes a list of tool requests against the provided clients."""
        tool_requests = [
            item for item in response.output if item.type == "function_call"
        ]
        tool_result_blocks: list[dict[str, str]] = []
        for tool_request in tool_requests:
            call_id = tool_request.call_id
            tool_name = tool_request.name

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    call_id,
                    json.dumps({"error": "Could not find that tool"}),
                )
                tool_result_blocks.append(tool_result_part)
                continue

            tool_output: CallToolResult | None = None
            try:
                tool_input = json.loads(tool_request.arguments)
                tool_output = await client.call_tool(
                    tool_name, tool_input
                )
                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(
                    {"error": content_list}
                    if tool_output and tool_output.isError
                    else content_list
                )
                tool_result_part = cls._build_tool_result_part(
                    call_id,
                    content_json,
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_part = cls._build_tool_result_part(
                    call_id,
                    json.dumps({"error": error_message}),
                )

            tool_result_blocks.append(tool_result_part)
        return tool_result_blocks
