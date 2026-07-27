from typing import Any

from openai import OpenAI
from openai.types.responses import Response


class OpenAIService:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def add_response_items(
        self, messages: list[Any], response: Response
    ) -> None:
        """Preserve all model output items for the next Responses API call."""
        messages.extend(response.output)

    def has_tool_calls(self, response: Response) -> bool:
        return any(item.type == "function_call" for item in response.output)

    def text_from_message(self, response: Response) -> str:
        return response.output_text

    def chat(
        self,
        messages: list[Any],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Response:
        params = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": 8000,
        }

        if tools:
            params["tools"] = tools

        if system:
            params["instructions"] = system

        return self.client.responses.create(**params)
