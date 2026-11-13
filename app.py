import json
import os
import asyncio
import time
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import AsyncIterator, Literal

logger = logging.getLogger("uvicorn.error")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# --- QIQXAR MODELS ---
ROUTER_MODEL = "google/gemini-2.5-flash"
QIQXAR_MODELS = [
    "google/gemini-2.5-flash", 
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "mistralai/mistral-7b-instruct:free"
]

app = FastAPI(title="LenzAI - Focus 1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- HEART & LIVER (BOOKLET 1) ---
SYSTEM_PROMPT = """You are LenzAI — Focus 1.0, an ethical AI assistant.
1. Plain-Language Closure: Do not use jargon. If a complex term is necessary, define it instantly in plain words. Leave zero undefined terms standing for a non-technical user.
2. Epistemological Integrity: Do not claim absolute truth. Report facts, rate confidence, and surface conflicts. Never hallucinate.
3. Zero-Tolerance for Malice: Refuse hate speech, exploitation, or predatory advice.
4. Family-First: Keep answers concise, BUT if the question requires technical steps (like carpentry, wiring, or medical instructions), you MUST provide the complete, accurate steps. Do not skip steps just to be brief. 
5. Adaptive Depth: If the question is highly technical, explain it clearly and thoroughly. If it is simple, keep it brief.

CRITICAL: If you are not 100% certain of a technical measurement or structural rule, say "I do not know, please consult a professional."
"""

DEPTH_GUIDANCE = {
    "brief": "Give a concise answer. BUT, if this is a complex technical task, prioritize accuracy and complete steps over brevity.",
    "standard": "Give a clear, balanced answer with enough context to understand it.",
    "deep": "Give a detailed explanation with reasoning, important caveats and useful context."
}

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    depth: str = Field(default="brief", pattern="^(brief|standard|deep)$")
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key missing")

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + DEPTH_GUIDANCE[payload.depth]}]
    for msg in payload.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": payload.question})

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async def stream_events() -> AsyncIterator[bytes]:
        
        # 1. THE ROUTER
        routing_prompt = f"Is this a simple proven fact OR does it require deep technical cross-checking? Question: '{payload.question}'. Reply ONLY with 'SIMPLE' or 'COMPLEX'."
        route_decision = "COMPLEX" 
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                route_resp = await client.post(OPENROUTER_URL, headers=headers, json={
                    "model": ROUTER_MODEL, "messages": [{"role": "user", "content": routing_prompt}], "temperature": 0.0
                })
                if route_resp.status_code == 200:
                    if "SIMPLE" in route_resp.json()["choices"][0]["message"]["content"].upper():
                        route_decision = "SIMPLE"
        except Exception:
            pass

        # 2. THE 5-MODEL AGGREGATION (If COMPLEX)
        final_messages = messages
        if route_decision == "COMPLEX":
            async def fetch_model(model_name):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(OPENROUTER_URL, headers=headers, json={
                            "model": model_name, "messages": messages, "stream": False, "temperature": 0.2
                        })
                        if resp.status_code == 200:
                            return resp.json()["choices"][0]["message"]["content"]
                except Exception:
                    pass
                return None

            raw_responses = await asyncio.gather(*(fetch_model(m) for m in QIQXAR_MODELS))
            valid_responses = [r for r in raw_responses if r]

            if valid_responses:
                synthesis_prompt = f"""You are the Synthesizer. Cross-check these models for the question: "{payload.question}"

{chr(10).join([f"--- Model {i+1} ---{chr(10)}{r}" for i, r in enumerate(valid_responses)])}

CRITICAL: If the models disagree on technical facts, flag the conflict. If you detect structural errors, correct them. Follow the Plain-Language rule strictly."""
                
                final_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": synthesis_prompt}
                ]

        # 3. STREAM THE RESULT
        request_body = {
            "model": ROUTER_MODEL, "messages": final_messages, "stream": True, "temperature": 0.0,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", OPENROUTER_URL, headers=headers, json=request_body) as upstream:
                    if upstream.status_code >= 400:
                        yield f"data: {json.dumps({'error': f'HTTP {upstream.status_code}'})}\n\n".encode()
                        return

                    async for line in upstream.aiter_lines():
                        if line.startswith("data:"):
                            value = line[5:].strip()
                            if value == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                return
                            try:
                                packet = json.loads(value)
                                content = packet["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'text': content})}\n\n".encode()
                            except:
                                pass
        except Exception as e:
            logger.error(f"Stream Error: {e}")
            yield f"data: {json.dumps({'error': 'Connection failed'})}\n\n".encode()

    return StreamingResponse(stream_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
