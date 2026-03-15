import asyncio
from unittest.mock import MagicMock
import numpy as np

class MockLiveSession:
    def __init__(self):
        self._queue = asyncio.Queue()

    async def send_realtime_input(self, audio=None, video=None):
        pass

    async def send_client_content(self, turns=None, turn_complete=True):
        # 1. Output transcription
        content1 = MagicMock()
        content1.model_turn = None
        content1.output_transcription.text = "This is a mocked response from Gemini Live."
        content1.turn_complete = False
        content1.interrupted = False
        
        msg1 = MagicMock()
        msg1.server_content = content1
        await self._queue.put(msg1)

        # 2. Turn complete
        content2 = MagicMock()
        content2.output_transcription.text = None
        content2.turn_complete = True
        content2.interrupted = False
        
        msg2 = MagicMock()
        msg2.server_content = content2
        await self._queue.put(msg2)

    async def receive(self):
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
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
        response.text = "This is a mocked response from the Gemini API (REST)."
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
        response.text = "This is a mocked response from the Gemini API (REST)."
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
