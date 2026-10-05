"""Local FastAPI server the Chrome extension talks to.

Run with: uvicorn server:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Show the memory pipeline's step-by-step logs in the uvicorn terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from memory_engine.pipeline import process_message, get_store
from memory_engine.retrieve import retrieve_relevant

app = FastAPI(title="Adaptive Memory Engine")

# Chrome extensions run as chrome-extension:// origins — allow all for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AddMessageRequest(BaseModel):
    conversation_id: str
    role: str  # "user" or "assistant" — currently informational, not yet used
    text: str


@app.post("/add_message")
def add_message(req: AddMessageRequest):
    result = process_message(req.conversation_id, req.text)
    return result


@app.get("/retrieve")
def retrieve(conversation_id: str, query: str, top_k: int = 5):
    store = get_store()
    items = store.get_all(conversation_id)
    relevant = retrieve_relevant(query, items, top_k=top_k)
    return {
        "results": [
            {
                "text": r["text"],
                "category": r["category"],
                "score": r["score"],
                "similarity": round(r["similarity"], 3),
            }
            for r in relevant
        ]
    }


@app.get("/export")
def export(conversation_id: str):
    store = get_store()
    return store.export_snapshot(conversation_id)


@app.delete("/conversation/{conversation_id}")
def clear_conversation(conversation_id: str):
    store = get_store()
    store.delete_conversation(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


@app.get("/health")
def health():
    return {"status": "ok"}
