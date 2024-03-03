import logging
from typing import Any, Dict, List, Union
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from .SimpleThrottle import SimpleThrottle
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

logger = logging.getLogger(__name__)

class AsyncStreamingSlackCallbackHandler(AsyncCallbackHandler):
    """Async callback handler for streaming to Slack. Only works with LLMs that support streaming."""

    def __init__(self, client: WebClient):
        self.client = client
        self.channel_id = None
        self.thread_ts = None
        self.update_delay = 0.1  # Set the desired delay in seconds
        self.update_throttle = SimpleThrottle(self._update_message_in_slack, self.update_delay)

    async def start_new_response(self, channel_id, thread_ts):
        """Initialize variables for a new response."""
        self.current_message = ""
        self.message_ts = None
        self.channel_id = channel_id
        self.thread_ts = thread_ts

    async def _update_message_in_slack(self):
        """Update the message in Slack."""
        try:
            await self.client.chat_update(
                channel=self.channel_id, ts=self.message_ts, text=self.current_message
            )
        except SlackApiError as e:
            print(f"Error updating message: {e}")

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle a new token from the LLM."""
        self.current_message += token
        await self.update_throttle.call()

    async def handle_llm_error(self, e: Exception) -> None:
        """Handle errors from the LLM."""
        try:
            logger.error(f"Got LLM Error. Will post about it: {e}")
            await self.client.chat_postMessage(text=str(e), channel=self.channel_id, thread_ts=self.thread_ts)
        except Exception as e:
            logger.error(f"Error posting exception message: {e}")

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Handle the start of the LLM."""
        try:
            if self.channel_id is None:
                raise Exception("channel_id is None")
            # Send an empty response and get the timestamp
            post_response = await self.client.chat_postMessage(text="...", channel=self.channel_id, thread_ts=self.thread_ts)
            self.message_ts: str = post_response["ts"]
        except Exception as e:
            await self.handle_llm_error(e)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle the end of the LLM."""
        try:
            await self.update_throttle.call_and_wait()
            # Make sure it got the last one:
            await self.start_new_response(self.channel_id, self.thread_ts)
        except Exception as e:
            await self.handle_llm_error(e)

    # ... (additional callback methods)

    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        print("Got text!", text)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
