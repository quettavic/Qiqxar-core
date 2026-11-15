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

ROUTER_MODEL = "openai/gpt-oss-20b"
SYNTH_MODEL = "openai/gpt-oss-120b"
QIQXAR_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

app = FastAPI(title="Qiqxar - Focus 1.0")

SYSTEM_PROMPT = """You are Qiqxar, an ethical AI assistant.
1. Plain-Language Closure: Do not use jargon. If a complex term is necessary, define it instantly in plain words. Leave zero undefined terms standing for a non-technical user.
2. Epistemological Integrity: Do not claim absolute truth. Report facts, rate confidence, and surface conflicts. Never hallucinate.
3. Zero-Tolerance for Malice: Refuse hate speech, exploitation, or predatory advice.
4. Family-First: Keep answers concise, BUT if the question requires technical steps, you MUST provide the complete, accurate steps. Do not skip steps just to be brief.
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Qiqxar</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-main: #131314;
            --bg-surface: #1e1f20;
            --bg-surface-hover: #282a2c;
            --user-pill: #282a2c;
            --text-primary: #e3e3e3;
            --text-secondary: #9aa0a6;
            --accent-blue: #a8c7fa;
            --accent-gradient: linear-gradient(135deg, #4285f4, #9b72cb, #d96570);
            --border-subtle: rgba(255, 255, 255, 0.08);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            display: flex;
            flex-direction: column;
            height: 100dvh;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 20px;
            background: var(--bg-main);
            z-index: 10;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 19px;
            font-weight: 500;
            color: var(--text-primary);
        }

        .brand-tag {
            font-size: 11px;
            color: var(--text-secondary);
            background: var(--bg-surface);
            padding: 2px 8px;
            border-radius: 10px;
            border: 1px solid var(--border-subtle);
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        select, .icon-btn {
            background: var(--bg-surface);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 6px 12px;
            font-size: 13px;
            cursor: pointer;
            outline: none;
        }

        select:hover, .icon-btn:hover {
            background: var(--bg-surface-hover);
        }

        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px 20px 32px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .canvas-column {
            width: 100%;
            max-width: 760px;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }

        #welcome-screen {
            margin-top: 12vh;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .greeting-title {
            font-size: 36px;
            font-weight: 500;
            line-height: 1.2;
        }

        .gradient-text {
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .greeting-sub {
            font-size: 22px;
            color: var(--text-secondary);
            font-weight: 400;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 8px;
        }

        .suggestion-chip {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 10px 16px;
            border-radius: 16px;
            font-size: 14px;
            cursor: pointer;
            text-align: left;
        }

        .suggestion-chip:hover {
            background: var(--bg-surface-hover);
        }

        .user-row {
            display: flex;
            justify-content: flex-end;
            width: 100%;
        }

        .user-bubble {
            background: var(--user-pill);
            color: var(--text-primary);
            padding: 12px 18px;
            border-radius: 22px;
            border-bottom-right-radius: 6px;
            max-width: 80%;
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .bot-row {
            display: flex;
            align-items: flex-start;
            gap: 14px;
            width: 100%;
        }

        .sparkle-icon {
            width: 26px;
            height: 26px;
            flex-shrink: 0;
            margin-top: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .bot-content {
            flex: 1;
            min-width: 0;
            font-size: 16px;
            line-height: 1.65;
            color: var(--text-primary);
            word-break: break-word;
        }

        .bot-content p { margin-bottom: 14px; }
        .bot-content p:last-child { margin-bottom: 0; }
        .bot-content strong { color: #ffffff; font-weight: 600; }
        .bot-content ul, .bot-content ol { margin: 10px 0 14px 22px; }
        .bot-content li { margin-bottom: 6px; }
        .bot-content code {
            background: var(--bg-surface);
            padding: 2px 6px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 14px;
            color: var(--accent-blue);
        }
        .bot-content pre {
            background: #18191a;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 14px;
            overflow-x: auto;
            margin: 14px 0;
        }
        .bot-content pre code { background: transparent; padding: 0; color: #e3e3e3; }

        .thinking-dots {
            color: var(--text-secondary);
            font-style: italic;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }

        #input-dock {
            padding: 10px 16px 20px 16px;
            background: var(--bg-main);
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .pill-box {
            width: 100%;
            max-width: 760px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 28px;
            padding: 8px 12px 8px 20px;
            display: flex;
            align-items: flex-end;
            gap: 8px;
        }

        .pill-box:focus-within {
            background: var(--bg-surface-hover);
            border-color: rgba(168, 199, 250, 0.4);
        }

        #user-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 16px;
            line-height: 1.5;
            max-height: 160px;
            min-height: 24px;
            resize: none;
            outline: none;
            font-family: inherit;
            padding: 6px 0;
        }

        #user-input::placeholder { color: var(--text-secondary); }

        .action-circle {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        #mic-btn { background: transparent; color: var(--text-primary); }
        #mic-btn:hover { background: rgba(255, 255, 255, 0.08); }
        #mic-btn.listening {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }

        #send-btn {
            background: var(--accent-blue);
            color: #131314;
            font-size: 18px;
            font-weight: bold;
        }
        #send-btn:disabled {
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-secondary);
        }

        .disclaimer {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span>Qiqxar</span>
            <span class="brand-tag">Focus 1.0</span>
        </div>
        <div class="header-actions">
            <select id="depth-select" title="Response Depth">
                <option value="brief">Brief</option>
                <option value="standard" selected>Standard</option>
                <option value="deep">Deep</option>
            </select>
            <button class="icon-btn" onclick="resetChat()" title="Start a new chat">+ New</button>
        </div>
    </header>

    <div id="chat-container">
        <div class="canvas-column" id="chat-box">
            <div id="welcome-screen">
                <div>
                    <div class="greeting-title"><span class="gradient-text">Hello.</span></div>
                    <div class="greeting-sub">How can Qiqxar help you today?</div>
                </div>
                <div class="chip-row">
                    <button class="suggestion-chip" onclick="useChip('Explain how multi-model consensus works in simple words')">How does consensus work?</button>
                    <button class="suggestion-chip" onclick="useChip('Give me a step-by-step checklist for framing a 10x10 shed')">10x10 shed framing steps</button>
                    <button class="suggestion-chip" onclick="useChip('Compare the pros and cons of magnesium glycinate vs citrate')">Magnesium types comparison</button>
                </div>
            </div>
        </div>
    </div>

    <div id="input-dock">
        <div class="pill-box">
            <textarea id="user-input" placeholder="Ask Qiqxar..." rows="1" onkeydown="handleKeyDown(event)"></textarea>
            <button id="mic-btn" class="action-circle" onclick="toggleMic()" title="Voice Input">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
            </button>
            <button id="send-btn" class="action-circle" onclick="sendMessage()" title="Send">↑</button>
        </div>
        <div class="disclaimer">Qiqxar cross-checks multiple AI models. Verify critical measurements and medical facts.</div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');
        const depthSelect = document.getElementById('depth-select');
        let history = [];

        const SPARKLE_SVG = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M12 2C12.3 7.2 16.8 11.7 22 12C16.8 12.3 12.3 16.8 12 22C11.7 16.8 7.2 12.3 2 12C7.2 11.7 11.7 7.2 12 2Z" fill="url(#qiqxarGrad)"/>
                <defs>
                    <linearGradient id="qiqxarGrad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#4285F4"/>
                        <stop offset="0.5" stop-color="#9B72CB"/>
                        <stop offset="1" stop-color="#D96570"/>
                    </linearGradient>
                </defs>
            </svg>
        `;

        // Safe Speech Recognition Wrapper
        let recognition = null;
        let isListening = false;
        
        try {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRecognition) {
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'en-US';

                recognition.onstart = () => {
                    isListening = true;
                    micBtn.classList.add('listening');
                    userInput.placeholder = "Listening...";
                };

                recognition.onresult = (event) => {
                    let transcript = "";
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        transcript += event.results[i][0].transcript;
                    }
                    userInput.value = transcript;
                    userInput.style.height = 'auto';
                    userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
                };

                recognition.onerror = () => { stopMicUI(); };
                recognition.onend = () => { stopMicUI(); };
            }
        } catch (e) {
            recognition = null;
        }

        function stopMicUI() {
            isListening = false;
            micBtn.classList.remove('listening');
            userInput.placeholder = "Ask Qiqxar...";
        }

        function toggleMic() {
            if (!recognition) {
                alert("Voice input requires a secure HTTPS connection or is not supported in this browser.");
                return;
            }
            if (isListening) {
                recognition.stop();
            } else {
                try {
                    recognition.start();
                } catch (e) {
                    recognition.stop();
                }
            }
        }

        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 160) + 'px';
        });

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function useChip(text) {
            userInput.value = text;
            sendMessage();
        }

        function resetChat() {
            history = [];
            chatBox.innerHTML = `
                <div id="welcome-screen">
                    <div>
                        <div class="greeting-title"><span class="gradient-text">Hello.</span></div>
                        <div class="greeting-sub">How can Qiqxar help you today?</div>
                    </div>
                    <div class="chip-row">
                        <button class="suggestion-chip" onclick="useChip('Explain how multi-model consensus works in simple words')">How does consensus work?</button>
                        <button class="suggestion-chip" onclick="useChip('Give me a step-by-step checklist for framing a 10x10 shed')">10x10 shed framing steps</button>
                        <button class="suggestion-chip" onclick="useChip('Compare the pros and cons of magnesium glycinate vs citrate')">Magnesium types comparison</button>
                    </div>
                </div>
            `;
        }

        function renderMarkdown(text) {
            if (typeof marked !== 'undefined') {
                return marked.parse(text);
            }
            return text;
        }

        async function sendMessage() {
            if (isListening && recognition) {
                try { recognition.stop(); } catch(e) {}
            }

            const text = userInput.value.trim();
            if (!text) return;

            const welcome = document.getElementById('welcome-screen');
            if (welcome) welcome.remove();

            appendUserMessage(text);
            userInput.value = '';
            userInput.style.height = 'auto';
            sendBtn.disabled = true;

            const botId = 'bot-' + Date.now();
            const botContentDiv = appendBotMessage('<span class="thinking-dots">Cross-checking models...</span>', botId);

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
                    botContentDiv.innerHTML = `<p><strong>Server Error (${response.status}):</strong> ${errData.detail || 'Check server logs.'}</p>`;
                    sendBtn.disabled = false;
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startswith('data: ')) {
                            const dataStr = line.slice(6).trim();
                            if (dataStr === '[DONE]') break;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.text) {
                                    fullText += data.text;
                                    botContentDiv.innerHTML = renderMarkdown(fullText);
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                } else if (data.error) {
                                    botContentDiv.innerHTML = `<p><strong>Error:</strong> ${data.error} (${data.detail || ''})</p>`;
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
                botContentDiv.innerHTML = `<p><strong>Connection Error:</strong> Could not reach the server.</p>`;
            }

            sendBtn.disabled = false;
            userInput.focus();
        }

        function appendUserMessage(text) {
            const row = document.createElement('div');
            row.className = 'user-row';
            const bubble = document.createElement('div');
            bubble.className = 'user-bubble';
            bubble.textContent = text;
            row.appendChild(bubble);
            chatBox.appendChild(row);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function appendBotMessage(initialHtml, id) {
            const row = document.createElement('div');
            row.className = 'bot-row';
            row.id = id;

            const icon = document.createElement('div');
            icon.className = 'sparkle-icon';
            icon.innerHTML = SPARKLE_SVG;

            const content = document.createElement('div');
            content.className = 'bot-content';
            content.innerHTML = initialHtml;

            row.appendChild(icon);
            row.appendChild(content);
            chatBox.appendChild(row);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return content;
        }
    </script>
</body>
</html>
    """

@app.post("/api/chat")
async def chat(payload: ChatRequest):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is missing on the server environment.")

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
                    content = route_resp.json()["choices"][0]["message"]["content"]
                    if "SIMPLE" in content.upper():
                        route_decision = "SIMPLE"
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
