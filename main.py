import json
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from httpx import request
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI
from contextlib import asynccontextmanager
from redis import asyncio as aioredis
from memory import RedisStorage
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)

class Settings(BaseSettings):
    openai_model: str = "deepseek-v4-flash"
    openai_api_key: str
    openai_base_url: str
    redis_url: str
    redis_password: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform any startup tasks here
    logging.info("Starting up the FastAPI application...")
    app.state.storage = RedisStorage(settings.redis_url, settings.redis_password)
    yield
    # Perform any shutdown tasks here
    logging.info("Shutting down the FastAPI application...")
    await app.state.storage.close()

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: Optional[str] = None
    base_url: Optional[str] = None

async def prepare_chat_context(request: ChatRequest, authorization: Optional[str] = Header(None)) -> Tuple[AsyncOpenAI, str, list[Dict[str,any]], RedisStorage]:
    # Get api_key from the Authorization header if provided, otherwise use the default from settings
    api_key = settings.openai_api_key
    if authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "").strip()
        else:
            raise HTTPException(status_code=401, detail="Invalid Authorization header format. Expected 'Bearer <API_KEY>'.")

    # Use the model and base_url from the request if provided, otherwise use the defaults
    model = request.model or settings.openai_model
    base_url = request.base_url or settings.openai_base_url

    if not model or not base_url:
        raise HTTPException(status_code=400, detail="Model and base_url must be provided either in the request or in the environment variables.")
    
    # Create an instance of AsyncOpenAI with the provided API key and base URL
    client = AsyncOpenAI(
        api_key=api_key, 
        base_url=base_url
    )

    # Get the RedisStorage instance from the app state
    storage: RedisStorage = app.state.storage

    # Retrieve the chat history for the given session_id from Redis
    history = await storage.get_history(request.session_id)
    messages = list(history)  # Create a copy of the history to avoid modifying the original list
    messages.append({"role": "user", "content": request.question})

    return client, model, messages, storage

@app.post("/chat")
# The `chat` endpoint allows users to send a question to the OpenAI API and receive a response. It accepts a `ChatRequest` object containing the question, and optionally, the model and base URL. The API key can be provided in the Authorization header as a Bearer token. If not provided, it defaults to the API key specified in the environment variables. The endpoint handles errors and returns appropriate HTTP status codes and messages.
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    client, model, messages, storage = await prepare_chat_context(request, authorization)
    try:
        # Make a request to the OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        assistant_answer = response.choices[0].message.content
        await storage.append_message(request.session_id, "user", request.question)
        await storage.append_message(request.session_id, "assistant", assistant_answer)
        return {"answer": assistant_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, authorization: Optional[str] = Header(None)):
    client, model, messages, storage = await prepare_chat_context(request, authorization)
    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer = ""
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_answer += content
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            if full_answer:
                await storage.append_message(request.session_id, "user", request.question)
                await storage.append_message(request.session_id, "assistant", full_answer)

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            logging.error(f"Streaming error for session {request.session_id}: {str(e)}")
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")