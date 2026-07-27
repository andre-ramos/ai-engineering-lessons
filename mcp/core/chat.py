from typing import Any

from core.claude import OpenAIService
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(
        self,
        openai_service: OpenAIService,
        clients: dict[str, MCPClient],
    ):
        self.openai_service = openai_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[Any] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.openai_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            self.openai_service.add_response_items(self.messages, response)

            if self.openai_service.has_tool_calls(response):
                partial_text = self.openai_service.text_from_message(response)
                if partial_text:
                    print(partial_text)
                tool_results = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                self.messages.extend(tool_results)
            else:
                final_text_response = self.openai_service.text_from_message(
                    response
                )
                break

        return final_text_response
