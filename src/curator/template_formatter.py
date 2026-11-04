"""
CuratorKIT Template Formatter for Cohere Models.
Serializes system prompts, tool schemas, user requests, and tool calls using native Cohere control tokens.
"""

from typing import Dict, List, Optional
import json


class CohereAgentFormatter:
    """
    Standardizes Cohere Command-R / Aya-Expanse control tokens and tool calling syntax.
    Native tokens:
      - <|START_OF_TURN_TOKEN|>
      - <|END_OF_TURN_TOKEN|>
      - <|SYSTEM_TOKEN|>
      - <|USER_TOKEN|>
      - <|CHATBOT_TOKEN|>
      - <|START_ACTION|>
      - <|END_ACTION|>
      - <|START_TOOL_RESULT|>
      - <|END_TOOL_RESULT|>
    """

    START_TURN = "<|START_OF_TURN_TOKEN|>"
    END_TURN = "<|END_OF_TURN_TOKEN|>"
    SYSTEM_TOKEN = "<|SYSTEM_TOKEN|>"
    USER_TOKEN = "<|USER_TOKEN|>"
    CHATBOT_TOKEN = "<|CHATBOT_TOKEN|>"
    START_ACTION = "<|START_ACTION|>"
    END_ACTION = "<|END_ACTION|>"
    START_TOOL_RESULT = "<|START_TOOL_RESULT|>"
    END_TOOL_RESULT = "<|END_TOOL_RESULT|>"

    DEFAULT_TOOL_DEFINITIONS = [
        {
            "name": "fetch_user_email",
            "description": "Fetches user emails from specified folder",
            "parameters": {
                "type": "object",
                "properties": {"folder": {"type": "string"}},
                "required": ["folder"],
            },
        },
        {
            "name": "send_wire_transfer",
            "description": "Executes authorized bank wire transfers",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_account": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["to_account", "amount"],
            },
        },
        {
            "name": "execute_bash_command",
            "description": "Runs shell commands in sandboxed environment",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    ]

    def __init__(self, tools: Optional[List[Dict]] = None):
        self.tools = tools or self.DEFAULT_TOOL_DEFINITIONS

    def build_system_header(self, custom_instructions: str = "") -> str:
        """Constructs the system header including tool specifications."""
        tools_str = "\n".join(json.dumps(t) for t in self.tools)
        header = (
            f"{self.START_TURN}{self.SYSTEM_TOKEN}"
            f"# Safety and Operational Guidelines\n"
            f"You are an authorized enterprise agent with tool execution capabilities.\n"
            f"{custom_instructions}\n"
            f"Available Tools:\n"
            f"{tools_str}\n"
            f"{self.END_TURN}"
        )
        return header

    def format_turn(self, role: str, content: str) -> str:
        """Formats a single conversational turn."""
        role_tag = {
            "system": self.SYSTEM_TOKEN,
            "user": self.USER_TOKEN,
            "chatbot": self.CHATBOT_TOKEN,
        }.get(role.lower(), self.USER_TOKEN)

        return f"{self.START_TURN}{role_tag}{content}{self.END_TURN}"

    def format_action_call(self, tool_name: str, arguments: Dict) -> str:
        """Formats tool dispatch syntax."""
        payload = json.dumps({"tool_name": tool_name, "parameters": arguments})
        return f"{self.START_ACTION}{payload}{self.END_ACTION}"
