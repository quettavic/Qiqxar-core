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

# Valid Groq production models
ROUTER_MODEL = "llama-3.1-8b-instant"
SYNTH_MODEL = "llama-3.3-70b-versatile"
QIQXAR_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
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

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

@app.get("/", response_class=HTMLResponse)
async def home():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qiqxar — No noise. Just information.</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
        
        header { background: #1e293b; border: 1px solid #475569; border-radius: 12px; padding: 12px 20px; margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .logo-area { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #0ea5e9, #6366f1); display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; color: white; box-shadow: 0 0 12px rgba(14, 165, 233, 0.4); }
        h1 { font-size: 1.2rem; font-weight: 600; margin: 0; }
        .tagline { font-size: 0.75rem; color: #94a3b8; }

        #chat-box { flex: 1; background: #1e293b; border-radius: 12px; padding: 20px; overflow-y: auto; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display: flex; flex-direction: column; gap: 15px; }
        .message-wrapper { display: flex; flex-direction: column; max-width: 85%; }
        .message-wrapper.user { align-self: flex-end; }
        .message-wrapper.bot { align-self: flex-start; }

        .message { padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 16px; white-space: pre-wrap; }
        .user .message { background: #0284c7; color: white; border-bottom-right-radius: 4px; }
        .bot .message { background: #334155; color: #f8fafc; border-bottom-left-radius: 4px; }
        
        .bot-actions { display: flex; gap: 8px; margin-top: 6px; margin-left: 8px; align-items: center; }
        .action-icon-btn { background: transparent; border: none; color: #94a3b8; cursor: pointer; font-size: 13px; padding: 4px 8px; border-radius: 6px; display: flex; align-items: center; gap: 4px; transition: background 0.2s, color 0.2s; }
        .action-icon-btn:hover { background: rgba(255,255,255,0.08); color: #0ea5e9; }
        .action-icon-btn.active { color: #0ea5e9; }

        #controls { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
        #user-input { flex: 1; padding: 15px; border: 1px solid #475569; border-radius: 10px; font-size: 16px; background: #1e293b; color: white; outline: none; }
        #user-input:focus { border-color: #0ea5e9; }
        
        button { padding: 15px 25px; background: #0ea5e9; color: white; border: none; border-radius: 10px; font-size: 16px; font-weight: bold; cursor: pointer; }
        button:disabled { background: #475569; cursor: not-allowed; }
        
        .icon-btn { background: #1e293b; border: 1px solid #475569; color: #94a3b8; padding: 15px; border-radius: 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        .icon-btn:hover { border-color: #0ea5e9; color: white; }
        .icon-btn.recording { border-color: #ef4444; color: #ef4444; background-color: rgba(239, 68, 68, 0.15); animation: pulse 1.5s infinite; }
        
        .header-btn { background: #0f172a; border: 1px solid #475569; color: #cbd5e1; padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: normal; cursor: pointer; transition: all 0.2s; }
        .header-btn:hover { border-color: #0ea5e9; color: white; }

        .disclaimer { font-size: 0.72rem; color: #64748b; text-align: center; }

        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <div class="logo-icon">Q</div>
            <div>
                <h1>Qiqxar</h1>
                <div class="tagline">No other way. Just information.</div>
            </div>
        </div>
        <button class="header-btn" onclick="exportChat()" title="Download Chat Transcript">📥 Download Chat</button>
    </header>

    <div id="chat-box">
        <div class="message-wrapper bot">
            <div class="message"><strong>Qiqxar:</strong> System online. How can I help you analyze your data today?</div>
        </div>
    </div>
    
    <div id="controls">
        <button id="micBtn" class="icon-btn" onclick="toggleMic()" title="Voice Command">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v1a7 7 0 0 1-14 0v-1"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
        </button>

        <input type="text" id="user-input" placeholder="Ask Qiqxar anything or use mic..." onkeypress="handleKeyPress(event)">
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <div class="disclaimer">Qiqxar can make mistakes. Verify important information.</div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('micBtn');
        let history = [];
        let recognition = null;
        let isRecording = false;
        let shouldAutoSend = false;

        function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }

        function toggleMic() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                alert("Speech recognition is not supported on this browser.");
                return;
            }

            if (isRecording) {
                recognition.stop();
                return;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            shouldAutoSend = false;

            recognition.onstart = function() {
                isRecording = true;
                micBtn.classList.add('recording');
            };

            recognition.onresult = function(event) {
                const speechResult = event.results[0][0].transcript;
                userInput.value = speechResult;
                shouldAutoSend = true;
            };

            recognition.onerror = function(event) {
                console.error("Speech recognition error:", event.error);
                shouldAutoSend = false;
                stopMicState();
            };

            recognition.onend = function() {
                stopMicState();
                if (shouldAutoSend && userInput.value.trim() !== '') {
                    shouldAutoSend = false;
                    sendMessage();
                }
            };

            recognition.start();
        }

        function stopMicState() {
            isRecording = false;
            micBtn.classList.remove('recording');
        }

        async function sendMessage() {
            if (isRecording && recognition) {
                recognition.stop();
                return;
            }

            const text = userInput.value.trim();
            if (!text) return;
            
            appendMessage('user', text);
            userInput.value = '';
            sendBtn.disabled = true;

            const typingId = 'typing-' + Date.now();
            appendMessage('bot', '<em>Thinking...</em>', typingId, false);

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        question: text, 
                        history: history 
                    })
                });

                document.getElementById(typingId)?.closest('.message-wrapper').remove();
                const botMsgId = 'bot-' + Date.now();
                appendMessage('bot', '', botMsgId, true);
                const botDiv = document.getElementById(botMsgId);

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
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6);
                            if (dataStr === '[DONE]') break;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.text) {
                                    fullText += data.text;
                                    botDiv.innerHTML = `<strong>Qiqxar:</strong> ${fullText}`;
                                    chatBox.scrollTop = chatBox.scrollHeight;
                                } else if (data.error) {
                                    botDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
                                }
                            } catch (e) {}
                        }
                    }
                }
                
                history.push({ role: "user", content: text });
                history.push({ role: "assistant", content: fullText });

            } catch (error) {
                document.getElementById(typingId)?.closest('.message-wrapper')?.remove();
                appendMessage('bot', '<strong>System Error:</strong> Could not reach the server.', null, false);
            }
            sendBtn.disabled = false;
            userInput.focus();
        }

        function appendMessage(sender, text, id = null, showActions = false) {
            const wrapper = document.createElement('div');
            wrapper.className = `message-wrapper ${sender}`;

            const msgDiv = document.createElement('div');
            msgDiv.className = `message`;
            if (id) msgDiv.id = id;
            msgDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'Qiqxar'}</strong> ${text}`;
            wrapper.appendChild(msgDiv);

            if (showActions && sender === 'bot') {
                const actionsDiv = document.createElement('div');
                actionsDiv.className = 'bot-actions';
                
                const copyBtn = document.createElement('button');
                copyBtn.className = 'action-icon-btn';
                copyBtn.innerHTML = '📋 Copy';
                copyBtn.onclick = () => {
                    const cleanText = msgDiv.innerText.replace('Qiqxar:', '').trim();
                    navigator.clipboard.writeText(cleanText);
                    copyBtn.innerHTML = '✓ Copied';
                    setTimeout(() => copyBtn.innerHTML = '📋 Copy', 2000);
                };

                const speakBtn = document.createElement('button');
                speakBtn.className = 'action-icon-btn';
                speakBtn.innerHTML = '🔊 Speak';
                speakBtn.onclick = () => {
                    if ('speechSynthesis' in window) {
                        window.speechSynthesis.cancel();
                        const cleanText = msgDiv.innerText.replace('Qiqxar:', '').trim();
                        const utterance = new SpeechSynthesisUtterance(cleanText);
                        window.speechSynthesis.speak(utterance);
                    } else {
                        alert('Text-to-speech not supported on this browser.');
                    }
                };

                const upBtn = document.createElement('button');
                upBtn.className = 'action-icon-btn';
                upBtn.innerHTML = '👍';
                upBtn.onclick = () => {
                    upBtn.classList.toggle('active');
                    downBtn.classList.remove('active');
                };

                const downBtn = document.createElement('button');
                downBtn.className = 'action-icon-btn';
                downBtn.innerHTML = '👎';
                downBtn.onclick = () => {
                    downBtn.classList.toggle('active');
                    upBtn.classList.remove('active');
                };

                actionsDiv.appendChild(copyBtn);
                actionsDiv.appendChild(speakBtn);
                actionsDiv.appendChild(upBtn);
                actionsDiv.appendChild(downBtn);
                wrapper.appendChild(actionsDiv);
            }

            chatBox.appendChild(wrapper);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function exportChat() {
            if (history.length === 0) {
                alert("No conversation history to export yet.");
                return;
            }
            let transcript = "Qiqxar Chat Transcript\n=====================\n\n";
            history.forEach(item => {
                const role = item.role === 'user' ? 'You' : 'Qiqxar';
                transcript += `${role}: ${item.content}\n\n`;
            });

            const blob = new Blob([transcript], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Qiqxar-Chat-${new Date().toISOString().slice(0,10)}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
