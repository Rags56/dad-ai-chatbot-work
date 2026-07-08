import json
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent_core import agent_executor, ended_sessions, get_thread_id, reset_thread


def _extract_text(msg) -> str:
    content = msg.content if hasattr(msg, "content") else msg
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


app = FastAPI(title="Dad AI Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_ended: bool = False


@app.get("/health")
async def health():
    return {"status": "ok"}


def _build_message(req: ChatRequest) -> HumanMessage:
    content = []
    if req.image:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{req.image}"},
        })
    content.append({"type": "text", "text": req.message})
    return HumanMessage(content=content)


def _detect_session_ended(messages) -> bool:
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "end_session":
            return True
    return False


def _find_last_ai_text(messages) -> Optional[str]:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return _extract_text(m)
    return None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id in ended_sessions:
        return ChatResponse(
            response="This session has ended. Please start a new session.",
            session_ended=True,
        )

    msg = _build_message(req)
    config = {"configurable": {"thread_id": get_thread_id(req.session_id)}}

    result = await agent_executor.ainvoke(
        {"messages": [msg]},
        config=config,
    )

    if _detect_session_ended(result["messages"]):
        ended_sessions.add(req.session_id)

    text = _find_last_ai_text(result["messages"])
    return ChatResponse(
        response=text or "I couldn't process that request.",
        session_ended=req.session_id in ended_sessions,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if req.session_id in ended_sessions:
        async def immediate():
            yield f"data: {json.dumps({'done': True, 'session_ended': True})}\n\n"
        return StreamingResponse(immediate(), media_type="text/event-stream")

    msg = _build_message(req)
    config = {"configurable": {"thread_id": get_thread_id(req.session_id)}}

    async def generate():
        all_messages = []
        async for chunk in agent_executor.astream(
            {"messages": [msg]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_data = chunk[0]
                all_messages.append(chunk_data)
                if hasattr(chunk_data, "content") and chunk_data.content:
                    if isinstance(chunk_data.content, list):
                        for part in chunk_data.content:
                            if isinstance(part, dict) and "text" in part:
                                yield f"data: {json.dumps({'token': part['text']})}\n\n"
                            elif isinstance(part, str):
                                yield f"data: {json.dumps({'token': part})}\n\n"
                    elif isinstance(chunk_data.content, str):
                        yield f"data: {json.dumps({'token': chunk_data.content})}\n\n"

        ended = _detect_session_ended(all_messages)
        if ended:
            ended_sessions.add(req.session_id)
        yield f"data: {json.dumps({'done': True, 'session_ended': ended})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/session/reset")
async def reset_session(req: ChatRequest):
    ended_sessions.discard(req.session_id)
    reset_thread(req.session_id)
    return {"session_id": req.session_id, "reset": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
