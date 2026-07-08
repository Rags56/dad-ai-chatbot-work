import os
import uuid
import base64
import gradio as gr
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


def _detect_ended(messages) -> bool:
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "end_session":
            return True
    return False


def _last_ai_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return _extract_text(m)
    return "I couldn't process that."


def _build_content(text: str, files: list) -> list:
    content = []
    for f in (files or []):
        filepath = f if isinstance(f, str) else getattr(f, "name", str(f))
        if os.path.isfile(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            mime_map = {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
            mime = mime_map.get(ext, "image/jpeg")
            with open(filepath, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
    content.append({"type": "text", "text": text})
    return content


def _parse_message(message) -> tuple[str, list]:
    if isinstance(message, str):
        return message, []
    if isinstance(message, dict):
        return message.get("text", ""), message.get("files", [])
    if hasattr(message, "text"):
        return message.text, getattr(message, "files", []) or []
    return str(message), []


with gr.Blocks(title="Dad AI - IT Support Chatbot") as demo:
    session_id_state = gr.State(value=uuid.uuid4().hex)
    session_ended_state = gr.State(value=False)

    gr.Markdown(
        "# Dad AI - IT Support Chatbot\n"
        "First-line IT support for Abu Odeh. Describe your issue or upload a screenshot."
    )

    new_session_btn = gr.Button("+ New Session", visible=False, variant="primary")

    chatbot = gr.Chatbot(height=480)

    with gr.Column(visible=True) as input_col:
        textbox = gr.MultimodalTextbox(
            placeholder="Describe your tech issue or upload a screenshot...",
            file_types=["image"],
            show_label=False,
        )

    with gr.Column(visible=False) as ended_col:
        gr.Markdown(
            "### Thank you for using the chatbot!\n\n"
            "This support session has ended. To start a new issue, click **+ New Session** above."
        )

    def do_new_session():
        new_id = uuid.uuid4().hex
        reset_thread("gradio_session")
        ended_sessions.discard("gradio_session")
        return (
            [],
            new_id,
            False,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", interactive=True),
        )

    def do_respond(message, history, session_id, session_ended):
        if session_ended:
            yield (
                history, session_id, session_ended,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=""),
            )
            return

        text, files = _parse_message(message)

        if files:
            file_refs = []
            for f in files:
                fp = f if isinstance(f, str) else getattr(f, "name", str(f))
                if os.path.isfile(fp):
                    file_refs.append({"file": {"path": fp}})
            history.append({"role": "user", "content": text or "(image)", "files": file_refs} if file_refs else {"role": "user", "content": text})
        else:
            history.append({"role": "user", "content": text})

        yield (
            history, session_id, session_ended,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", interactive=False),
        )

        content = _build_content(text, files)
        msg = HumanMessage(content=content)
        config = {"configurable": {"thread_id": get_thread_id("gradio_session")}}

        result = agent_executor.invoke({"messages": [msg]}, config=config)
        response_text = _last_ai_text(result["messages"])

        ended = _detect_ended(result["messages"])
        if ended:
            ended_sessions.add("gradio_session")

        history.append({"role": "assistant", "content": response_text})

        yield (
            history, session_id, ended,
            gr.update(visible=ended),
            gr.update(visible=not ended),
            gr.update(visible=ended),
            gr.update(value="", interactive=not ended),
        )

    textbox.submit(
        do_respond,
        [textbox, chatbot, session_id_state, session_ended_state],
        [chatbot, session_id_state, session_ended_state, new_session_btn, input_col, ended_col, textbox],
    )

    new_session_btn.click(
        do_new_session,
        None,
        [chatbot, session_id_state, session_ended_state, new_session_btn, input_col, ended_col, textbox],
    )


if __name__ == "__main__":
    demo.launch(share=False)
