import pytest
from httpx import ASGITransport, AsyncClient
from main import app, lifespan

pytestmark = pytest.mark.anyio

@pytest.fixture(scope="function")
async def client():
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
            
async def test_chat_stream_endpoint(client):
    payload = {
        "session_id": "pytest_session_123",
        "question": "1+1等于几？"
    }
    headers = {}
    async with client.stream("POST", "/chat/stream", json=payload, headers=headers) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        chunks = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line)

        assert len(chunks) > 0, "应该接收到流式数据分片"
        full_response_text = "".join(chunks)
        assert "2" in full_response_text or "two" in full_response_text.lower()