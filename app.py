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
                <line x1="8" y1="23" x2="16" y2
