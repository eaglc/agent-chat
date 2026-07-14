from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI

import logging

logging.basicConfig(level=logging.INFO)

class Settings(BaseSettings):
    openai_model: str = "deepseek-v4-flash"
    openai_api_key: str
    openai_base_url: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = None
    base_url: Optional[str] = None

@app.post("/chat")
# The `chat` endpoint allows users to send a question to the OpenAI API and receive a response. It accepts a `ChatRequest` object containing the question, and optionally, the model and base URL. The API key can be provided in the Authorization header as a Bearer token. If not provided, it defaults to the API key specified in the environment variables. The endpoint handles errors and returns appropriate HTTP status codes and messages.
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):

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

    logging.info(f"Using model: {model}, base_url: {base_url} api_key: {api_key}")

    try:
        # Make a request to the OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.question}]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))