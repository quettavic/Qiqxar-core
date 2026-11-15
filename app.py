
import json
import os
import logging
import httpx

from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


logger = logging.getLogger("uvicorn.error")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL_NAME = "openai/gpt-oss-20b"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI(title="Qiqxar — No noise. Just information.")


# --------------------------------------------------
# STATIC FRONTEND
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


@app.get("/")
async def home():
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=500,
            detail="Frontend file not found: static/index.html"
        )

    return FileResponse(INDEX_FILE)


# --------------------------------------------------
# AI SYSTEM INSTRUCTIONS
# --------------------------------------------------

SYSTEM_PROMPT = """You are Qiqxar, an ethical AI assistant.

1. Plain-Language Closure:
Do not use jargon. If a complex term is necessary,
define it immediately in plain language.

2. Epistemological Integrity:
Do not claim absolute truth. Report facts,
confidence, uncertainty and conflicts honestly.
Never invent information or sources.

3. Responsible Assistance:
Do not assist with exploitation, abuse or
instructions intended to cause harm.

4. Complete Instructions:
Keep ordinary answers concise. When technical
steps are required, provide complete and accurate
instructions rather than skipping important steps.

5. Adaptive Depth:
Answer simple questions briefly.
Explain complex questions clearly and thoroughly.

6. Honest Limitations:
Never pretend to have checked another AI model,
searched the internet or independently verified
information unless that actually happened.

7. User Independence:
Present information clearly and let the user
make their own decisions.

For technical measurements or structural rules,
state uncertainty when appropriate and recommend
a qualified professional when safety requires it.
"""


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "online",
        "application": "Qiqxar",
        "provider": "Groq",
        "model": MODEL_NAME,
        "provider_configured": bool(API_KEY),
        "frontend_found": INDEX_FILE.is_file(),
        "verification_system": "not yet implemented"
    }


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat")
async def chat(payload: ChatRequest):

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is missing."
        )

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Preserve recent conversation context.
    for msg in payload.history[-20:]:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    messages.append({
        "role": "user",
        "content": question
    })

    request_body = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.2
    }

    async def stream_events() -> AsyncIterator[bytes]:

        try:
            async with httpx.AsyncClient(
                timeout=90.0
            ) as client:

                async with client.stream(
                    "POST",
                    GROQ_URL,
                    headers=headers,
                    json=request_body
                ) as upstream:

                    if upstream.status_code >= 400:

                        body = (
                            await upstream.aread()
                        ).decode(errors="ignore")[:1000]

                        error_data = {
                            "error": f"HTTP {upstream.status_code}",
                            "detail": body
                        }

                        yield (
                            f"data: {json.dumps(error_data)}\n\n"
                        ).encode()

                        return

                    async for line in upstream.aiter_lines():

                        if not line.startswith("data:"):
                            continue

                        value = line[5:].strip()

                        if value == "[DONE]":
                            yield b"data: [DONE]\n\n"
                            return

                        if not value:
                            continue

                        try:
                            packet = json.loads(value)

                            choices = packet.get("choices", [])

                            if not choices:
                                continue

                            delta = choices[0].get(
                                "delta", {}
                            )

                            content = delta.get(
                                "content", ""
                            )

                            if content:
                                event_data = {
                                    "text": content
                                }

                                yield (
                                    f"data: {json.dumps(event_data)}\n\n"
                                ).encode()

                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue

        except Exception as error:

            logger.exception("Qiqxar stream error")

            error_data = {
                "error": "Connection failed",
                "detail": str(error)
            }

            yield (
                f"data: {json.dumps(error_data)}\n\n"
            ).encode()

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
