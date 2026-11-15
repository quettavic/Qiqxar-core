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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qiqxar - Focus 1.0</title>
    <style>
        :root {
            --bg-main: #131314;
            --bg-chat: #1e1f20;
            --bg-input: #282a2c;
            --text-main: #e3e3e3;
            --text-muted: #8e918f;
            --accent: #a8c7fa;
            --user-bubble: #004a77;
            --bot-bubble: #1e1f20;
            --border: #333537;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: var(--bg-main); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        /* Top Navigation Header */
        header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid var(--border); background: var(--bg-main); }
        .logo-area { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 18px; letter-spacing: 0.5px; color: var(--text-main); }
        .logo-badge { background: linear-gradient(135deg, #4285f4, #9b72cb); color: white; font-size: 12px; padding: 2px 8px; border-radius: 12px; font-weight: 500; }
        
        /* Main Chat Area Container */
        #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; align-items: center; gap: 24px; scroll-behavior: smooth; }
        .chat-width-limiter { width: 100%; max-width: 800px; display: flex; flex-direction: column; gap: 24px; }
        
        /* Message Rows */
        .message-row { display: flex; gap: 16px; width: 100%; animation: fadeIn 0.3s ease; }
        .message-row.user { justify-content: flex-end; }
        .message-row.bot { justify-content: flex-start; }
        
        .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; flex-shrink: 0; }
        .user-avatar { background: #005fb8; color: white; }
        .bot-avatar { background: #2d3748; color: #a8c7fa; border: 1px solid var(--border); }
        
        .message-bubble { max-width: 75%; padding: 14px 18px; border-radius: 16px; line-height: 1.6; font-size: 15px; word-break: break-word; white-space: pre-wrap; }
        .message-row.user .message-bubble { background: var(--user-bubble); color: white; border-top-right-radius: 4px; }
        .message-row.bot .message-bubble { background: var(--bot-bubble); color: var(--text-main); border: 1px solid var(--border); border-top-left-radius: 4px; }
        
        /* Input Footer Box */
        #input-area { padding: 16px 24px 24px 24px; background: var(--bg-main); display: flex; justify-content: center; }
        .input-wrapper { width: 100%; max-width: 800px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 24px; padding: 10px 16px; display: flex; flex-direction: column; gap: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: border-color 0.2s; }
        .input-wrapper:focus-within { border-color: var(--accent); }
        
        #user-input { width: 100%; background: transparent; border: none; color: var(--text-main); font-size: 15px; outline: none; resize: none; max-height: 150px; min-height: 24px; font-family: inherit; }
        
        .input-controls { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px; }
        
        select { background: transparent; border: none; color: var(--text-muted); font-size: 13px; outline: none; cursor: pointer; padding: 4px 8px; border-radius: 6px; }
        select:hover { background: rgba(255,255,255,0.05); color: var(--text-main); }
        
        #send-btn { background: var(--accent); color: #111; border: none; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; font-weight: bold; cursor: pointer; transition: transform 0.1s, opacity 0.2s; }
        #send-btn:hover { transform: scale(1.05); }
        #send-btn:disabled { background: var(--border); color: var(--text-muted); cursor: not-allowed; transform: none; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <span>Qiqxar</span>
            <span class="logo-badge">Focus 1.0</span>
        </div>
        <select id="depth-select">
            <option value="brief">Depth: Brief</option>
            <option value="standard" selected>Depth: Standard</option>
            <option value="deep">Depth: Deep</option>
        </select>
    </header>

    <div id="chat-container">
        <div class="chat-width-limiter" id="chat-box">
            <div class="message-row bot">
                <div class="avatar bot-avatar">Q</div>
                <div class="message-bubble">Hello. I am Qiqxar. How can I help you today?</div>
            </div>
        </div>
    </div>

    <div id="input-area">
        <div class="input-wrapper">
            <textarea id="user-input" placeholder="Ask Qiqxar anything..." rows="1" onkeydown="handleKeyDown(event)"></textarea>
            <div class="input-controls">
                <span style="font-size: 12px; color: var(--text-muted);">Multi-LLM Consensus Active</span>
                <button id="send-btn" onclick="sendMessage()" title="Send Message">↑</button>
            </div>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const depthSelect = document.getElementById('depth-select');
        let history = [];

        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            
            appendMessage('user', text);
            userInput.value = '';
            userInput.style.height = 'auto';
            sendBtn.disabled = true;

            const typingId = 'typing-' + Date.now();
            appendMessage('bot', 'Thinking...', typingId, true);

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
                    appendMessage('bot', `Server Error (${response.status}): ${errData.detail || 'Check API configuration.'}`);
                    sendBtn.disabled = false;
                    return;
                }

                document.getElementById(typingId).remove();
                const botMsgId = 'bot-' + Date.now();
                appendMessage('bot', '', botMsgId);
                const botBubble = document.getElementById(botMsgId).querySelector('.message-bubble');

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6).trim();
                            if (dataStr === '[DONE]') break;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.text) {
                                    fullText += data.text;
                                    botBubble.textContent = fullText;
                                    scrollToBottom();
                                } else if (data.error) {
                                    botBubble.textContent = `Error: ${data.error} (${data.detail || ''})`;
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
                appendMessage('bot', 'System Error: Could not reach the server.');
            }
            sendBtn.disabled = false;
            userInput.focus();
        }

        function appendMessage(sender, text, id = null, isHtml = false) {
            const row = document.createElement('div');
            row.className = `message-row ${sender}`;
            if (id) row.id = id;

            const avatar = document.createElement('div');
            avatar.className = `avatar ${sender}-avatar`;
            avatar.textContent = sender === 'user' ? 'U' : 'Q';

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            if (isHtml) {
                bubble.innerHTML = `<em>${text}</em>`;
            } else {
                bubble.textContent = text;
            }

            if (sender === 'user') {
                row.appendChild(bubble);
                row.appendChild(avatar);
            } else {
                row.appendChild(avatar);
                row.appendChild(bubble);
            }

            chatBox.appendChild(row);
            scrollToBottom();
        }

        function scrollToBottom() {
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
    </script>
</body>
</html>
    """
