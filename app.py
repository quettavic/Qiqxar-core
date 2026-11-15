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

# 100% verified, universally accessible Groq models
ROUTER_MODEL = "llama-3.1-8b-instant"
SYNTH_MODEL = "llama-3.1-8b-instant"
QIQXAR_MODELS = [
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Qiqxar — No noise. Just information.</title>
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

        .icon-btn {
            background: var(--bg-surface);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 8px 14px;
            font-size: 13px;
            cursor: pointer;
            outline: none;
            transition: background 0.2s;
        }

        .icon-btn:hover {
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
            transition: background 0.2s;
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
            flex-direction: column;
            gap: 8px;
            width: 100%;
        }

        .bot-main-content {
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

        .bot-actions { display: flex; gap: 8px; margin-left: 40px; align-items: center; }
        .action-icon-btn { background: transparent; border: none; color: var(--text-secondary); cursor: pointer; font-size: 13px; padding: 4px 8px; border-radius: 6px; display: flex; align-items: center; gap: 4px; transition: background 0.2s, color 0.2s; }
        .action-icon-btn:hover { background: var(--bg-surface-hover); color: var(--accent-blue); }
        .action-icon-btn.active { color: var(--accent-blue); }

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
        #mic-btn.recording {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            animation: pulse 1.5s infinite;
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
            <button class="icon-btn" onclick="exportChat()" title="Download Chat Transcript">📥 Download Chat</button>
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
            <button id="mic-btn" class="action-circle" onclick="toggleMic()" title="Voice Command">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
            </button>
            <button id="send-btn" class="action-circle" onclick="sendMessage()" title="Send">↑</button>
        </div>
        <div class="disclaimer">Qiqxar can make mistakes. Verify important information.</div>
    </div>

    <script>
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

        let history = [];
        let recognition = null;
        let isRecording = false;
        let shouldAutoSend = false;

        try {
            const SpeechAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechAPI) {
                recognition = new SpeechAPI();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                recognition.lang = 'en-US';

                recognition.onstart = function() {
                    isRecording = true;
                    const mic = document.getElementById('mic-btn');
                    if (mic) mic.classList.add('recording');
                    const input = document.getElementById('user-input');
                    if (input) input.placeholder = "Listening...";
                };

                recognition.onresult = function(event) {
                    const speechResult = event.results[0][0].transcript;
                    const input = document.getElementById('user-input');
                    if (input) {
                        input.value = speechResult;
                        input.style.height = 'auto';
                        input.style.height = Math.min(input.scrollHeight, 160) + 'px';
                    }
                    shouldAutoSend = true;
                };

                recognition.onerror = function(event) {
                    console.error("Speech recognition error:", event.error);
                    shouldAutoSend = false;
                    stopMicUI();
                };

                recognition.onend = function() {
                    stopMicUI();
                    if (shouldAutoSend && document.getElementById('user-input').value.trim() !== '') {
                        shouldAutoSend = false;
                        sendMessage();
                    }
                };
            }
        } catch (e) {
            recognition = null;
        }

        function stopMicUI() {
            isRecording = false;
            const mic = document.getElementById('mic-btn');
            if (mic) mic.classList.remove('recording');
            const input = document.getElementById('user-input');
            if (input) input.placeholder = "Ask Qiqxar...";
        }

        function toggleMic() {
            if (!recognition) {
                alert("Speech recognition is not supported on this browser.");
                return;
            }
            if (isRecording) {
                recognition.stop();
            } else {
                shouldAutoSend = false;
                recognition.start();
            }
        }

        document.addEventListener('input', function(e) {
            if (e.target && e.target.id === 'user-input') {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
            }
        });

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function useChip(text) {
            const input = document.getElementById('user-input');
            if (input) {
                input.value = text;
                sendMessage();
            }
        }

        function resetChat() {
            history = [];
            const chatBox = document.getElementById('chat-box');
            if (chatBox) {
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
        }

        function renderMarkdown(text) {
            try {
                if (typeof marked !== 'undefined' && marked.parse) {
                    return marked.parse(text);
                }
            } catch (e) {}
            return text;
        }

        async function sendMessage() {
            if (isRecording && recognition) {
                try { recognition.stop(); } catch(e) {}
            }

            const input = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const chatBox = document.getElementById('chat-box');
            const chatContainer = document.getElementById('chat-container');

            if (!input || !sendBtn || !chatBox) return;

            const text = input.value.trim();
            if (!text) return;

            const welcome = document.getElementById('welcome-screen');
            if (welcome) welcome.remove();
