import json
import os
import asyncio
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import AsyncIterator, Literal

logger = logging.getLogger("uvicorn.error")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("GROQ_API_KEY", "").strip()

ROUTER_MODEL = "llama-3.1-8b-instant"
SYNTH_MODEL = "llama-3.3-70b-versatile"
QIQXAR_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

app = FastAPI(title="LenzAI - Focus 1.0")

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

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LenzAI - Focus 1.0</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
        #chat-box { flex: 1; background: #1e293b; border-radius: 12px; padding: 20px; overflow-y: auto; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .message { margin-bottom: 15px; padding: 12px 16px; border-radius: 18px; max-width: 85%; line-height: 1.5; font-size: 16px; white-space: pre-wrap; }
        .user { background: #0284c7; color: white; align-self: flex-end; margin-left: auto; border-bottom-right-radius: 4px; }
        .bot { background: #334155; color: #f8fafc; align-self: flex-start; border-bottom-left-radius: 4px; }
        #controls { display: flex; gap: 10px; align-items: center; }
        #user-input { flex: 1; padding: 15px; border: 1px solid #475569; border-radius: 10px; font-size: 16px; background: #1e293b; color: white; outline: none; }
        button { padding: 15px 25px; background: #0ea5e9; color: white; border: none; border-radius: 10px; font-size: 16px; font-weight: bold; cursor: pointer; }
        button:disabled { background: #475569; }
        select { padding: 12px; border-radius: 10px; border: 1px solid #475569; background: #1e293b; color: white; font-size: 14px; }
    </style>
</head>
<body>
    <div id="chat-box">
        <div class="message bot"><strong>LenzAI:</strong> Hello. I am LenzAI. (Experimental, AS-IS). How can I help you today?</div>
    </div>
    <div id="controls">
        <select id="depth-select">
            <option value="brief">Brief</option>
            <option value="standard">Standard</option>
            <option value="deep">Deep</option>
        </select>
        <input type="text" id="user-input" placeholder="Ask LenzAI anything..." onkeypress="handleKeyPress(event)">
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const depthSelect = document.getElementById('depth-select');
        let history = [];

        function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            
            appendMessage('user', text);
            userInput.value = '';
            sendBtn.disabled = true;

            const typingId = 'typing-' + Date.now();
            appendMessage('bot', '<em>Thinking...</em>', typingId);

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        question: text, 
                        depth: depthSelect.value,
                        history: history 
                    })
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    document.getElementById(typingId)?.remove();
                    appendMessage('bot', `<strong>Server Error (${response.status}):</strong> ${errData.detail || 'Check your GROQ_API_KEY.'}`);
                    sendBtn.disabled = false;
                    return;
                }

                document.getElementById(typingId).remove();
                const botMsgId = 'bot-' + Date.now();
                appendMessage('bot', '', botMsgId);
                const botDiv = document.getElementById(botMsgId);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6).trim();
                            if (dataStr === '[DONE]') break;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.text) {
                                    fullText += data.text;
                                    botDiv.innerHTML = `<strong>LenzAI:</strong> ${fullText}`;
                                    chatBox.scrollTop = chatBox.scrollHeight;
                                } else if (data.error) {
                                    botDiv.innerHTML = `<strong>Error:</strong> ${data.error} (${data.detail || ''})`;
                                }
                            } catch (e) {}
                        }
                    }
                }
                
                if (fullText) {
                    history.push({ role: "user", content: text });
                    history.push({ role: "assistant", content: fullText });
                }

            } catch (error) {
                document.getElementById(typingId)?.remove();
                appendMessage('bot', '<strong>System Error:</strong> Could not reach the server.');
            }
            sendBtn.disabled = false;
            userInput.focus();
        }

        function appendMessage(sender, text, id = null) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${sender}`;
            if (id) msgDiv.id = id;
            msgDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'LenzAI'}:</strong> ${text}`;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
    """

@app.post("/api/chat")
async def chat(payload: ChatRequest):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is missing on the server.")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + DEPTH_GUIDANCE[payload.depth]}]
    for msg in payload.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": payload.question})

    async def stream_events() -> AsyncIterator[bytes]:

        route_decision = "COMPLEX"
        routing_prompt = (
            f"Is this a simple proven fact OR does it require deep technical cross-checking? "
            f"Question: '{payload.question}'. Reply ONLY with 'SIMPLE' or 'COMPLEX'."
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                route_resp = await client.post(GROQ_URL, headers=headers, json={
                    "model": ROUTER_MODEL,
                    "messages": [{"role": "user", "content": routing_prompt}],
                    "temperature": 0.0,
                })
                if route_resp.status_code == 200:
                    if "SIMPLE" in route_resp.json()["choices"][0]["message"]["content"].upper():
                        route_decision = "SIMPLE"
                else:
                    logger.warning(f"Router HTTP {route_resp.status_code}: {route_resp.text[:300]}")
        except Exception as e:
            logger.warning(f"Router exception: {e}")

        final_messages = messages
        if route_decision == "COMPLEX":
            async def fetch_model(model_name: str):
                try:
                    async with httpx.AsyncClient(timeout=45.0) as client:
                        resp = await client.post(GROQ_URL, headers=headers, json={
                            "model": model_name,
                            "messages": messages,
                            "stream": False,
                            "temperature": 0.2,
                        })
                        if resp.status_code == 200:
                            return resp.json()["choices"][0]["message"]["content"]
                        logger.warning(f"{model_name} -> HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    logger.warning(f"{model_name} exception: {e}")
                return None

            raw_responses = await asyncio.gather(*(fetch_model(m) for m in QIQXAR_MODELS))
            valid_responses = [r for r in raw_responses if r]

            if valid_responses:
                synthesis_prompt = (
                    f'You are the Synthesizer. Cross-check these models for the question: "{payload.question}"\n\n'
                    + "\n".join(
                        f"--- Model {i+1} ---\n{r}" for i, r in enumerate(valid_responses)
                    )
                    + "\n\nCRITICAL: If the models disagree on technical facts, flag the conflict. "
                      "If you detect structural errors, correct them. Follow the Plain-Language rule strictly."
                )
                final_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": synthesis_prompt},
                ]

        request_body = {
            "model": SYNTH_MODEL,
            "messages": final_messages,
            "stream": True,
            "temperature": 0.0,
        }

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream("POST", GROQ_URL, headers=headers, json=request_body) as upstream:
                    if upstream.status_code >= 400:
                        body = (await upstream.aread()).decode(errors="ignore")[:500]
                        yield f"data: {json.dumps({'error': f'HTTP {upstream.status_code}', 'detail': body})}\n\n".encode()
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
                            except Exception:
                                pass
        except Exception as e:
            logger.error(f"Stream Error: {e}")
            yield f"data: {json.dumps({'error': 'Connection failed', 'detail': str(e)})}\n\n".encode()

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
