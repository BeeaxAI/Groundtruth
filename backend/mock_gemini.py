"""
Mock Gemini client for local testing without a real API key.
All content attributes are explicitly set to avoid MagicMock
truthy-default issues that crash JSON serialization.
"""

import asyncio
from unittest.mock import MagicMock
import numpy as np


def _make_server_content(
    model_turn=None,
    output_text=None,
    input_text=None,
    turn_complete=False,
    interrupted=False,
):
    """
    Build a properly structured server_content mock.

    Every attribute that the handler checks MUST be explicitly set,
    otherwise MagicMock defaults to truthy, causing:
      - handler enters wrong code path
      - MagicMock gets JSON-serialized → TypeError → crash → reconnect
    """
    content = MagicMock()
    content.model_turn = model_turn
    content.turn_complete = turn_complete
    content.interrupted = interrupted

    # output_transcription
    if output_text is not None:
        content.output_transcription = MagicMock()
        content.output_transcription.text = output_text
    else:
        content.output_transcription = None

    # input_transcription
    if input_text is not None:
        content.input_transcription = MagicMock()
        content.input_transcription.text = input_text
    else:
        content.input_transcription = None

    return content


class MockLiveSession:
    def __init__(self):
        self._queue = asyncio.Queue()

    async def send_realtime_input(self, audio=None, video=None):
        pass

    async def send_client_content(self, turns=None, turn_complete=True):
        # 1. Output transcription (what Gemini says)
        content1 = _make_server_content(
            output_text="This is a mocked response from Gemini Live.",
        )
        msg1 = MagicMock()
        msg1.server_content = content1
        await self._queue.put(msg1)

        # 2. Turn complete signal
        content2 = _make_server_content(turn_complete=True)
        msg2 = MagicMock()
        msg2.server_content = content2
        await self._queue.put(msg2)

    async def receive(self):
        while True:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                yield item
            except asyncio.TimeoutError:
                # Yield empty heartbeat — server_content=None gets skipped
                empty = MagicMock()
                empty.server_content = None
                yield empty

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockModels:
    """Synchronous models client."""

    def generate_content(self, model, contents, config=None):
        response = MagicMock()
        response.text = (
            "This is a mocked response from the Gemini API (REST)."
        )
        return response

    def embed_content(self, model, contents, task_type=None, title=None):
        if isinstance(contents, str):
            contents = [contents]
        response = MagicMock()
        response.embeddings = [
            MagicMock(values=np.zeros(768).tolist()) for _ in contents
        ]
        return response


class MockAioModels:
    """Asynchronous models client."""

    async def generate_content(self, model, contents, config=None):
        response = MagicMock()
        response.text = (
            "This is a mocked response from the Gemini API (REST)."
        )
        return response

    async def embed_content(self, model, contents, task_type=None, title=None):
        if isinstance(contents, str):
            contents = [contents]
        response = MagicMock()
        response.embeddings = [
            MagicMock(values=np.zeros(768).tolist()) for _ in contents
        ]
        return response


class MockAio:
    def __init__(self):
        self.live = MagicMock()
        self.live.connect.return_value = MockLiveSession()
        self.models = MockAioModels()


class MockGeminiClient:
    def __init__(self, api_key=None):
        self.aio = MockAio()
        self.models = MockModels()
