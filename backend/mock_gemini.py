import asyncio
from unittest.mock import MagicMock

class MockLiveSession:
    def __init__(self):
        pass

    async def send_realtime_input(self, audio=None, video=None):
        pass

    async def send_client_content(self, turns=None, turn_complete=True):
        pass

    async def receive(self):
        # Keeps the connection alive for tests
        while True:
            await asyncio.sleep(1)
            # Yield empty content to keep loop running but idle
            yield MagicMock(server_content=None)

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

class MockModels:
    def generate_content(self, model, contents, config=None):
        return MagicMock(text="This is a mocked response from the Gemini API.")

class MockAio:
    def __init__(self):
        self.live = MagicMock()
        self.live.connect.return_value = MockLiveSession()

class MockGeminiClient:
    def __init__(self, api_key=None):
        self.aio = MockAio()
        self.models = MockModels()
