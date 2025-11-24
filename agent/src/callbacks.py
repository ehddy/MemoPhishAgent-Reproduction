from collections import defaultdict
import logging

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler

logger = logging.getLogger(__name__)


class ToolUsageTracker(BaseCallbackHandler):
    """Count how many times each tool is invoked."""

    def __init__(self):
        self.counts = defaultdict(int)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """
        Called whenever the agent is about to invoke a tool.
        `serialized["name"]` is the tool name.
        """
        tool_name = serialized.get("name")
        logging.debug(f"Trying to get tool name: {tool_name}")
        if tool_name:
            self.counts[tool_name] += 1


class LLMCounter(BaseCallbackHandler):
    """Count how many times the LLM is called (across all prompts)."""

    def __init__(self):
        self.count = 0

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        """
        Called whenever the LLM is about to be invoked.
        We simply increment by one per call.
        """
        self.count += 1


def get_default_callbacks() -> list[BaseCallbackHandler]:
    """
    Return the list of callbacks to attach to every LLM.
    """
    return [
        ToolUsageTracker(),
        LLMCounter(),
    ]


def get_token_usage_callbacks() -> UsageMetadataCallbackHandler:
    """
    Return the callback for tracking token usage.
    """
    return UsageMetadataCallbackHandler()
