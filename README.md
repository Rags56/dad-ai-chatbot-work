# Dad AI — IT Support Chatbot

AI-powered first-line IT support chatbot for Abu Odeh (fictional company). Built with LangChain, Google Gemini, and Gradio.

## What it does

- Answers tech support questions (Outlook, Windows 11, VPN, Chrome, Office 365, etc.)
- Searches the web via DuckDuckGo for troubleshooting steps
- Creates support tickets when issues can't be resolved
- Supports text + image input (upload a screenshot of your error)
- Ends the session cleanly with a thank-you banner when the problem is solved

## Setup

### 1. Prerequisites

- Python 3.10 or newer
- A [Google AI API key](https://aistudio.google.com/apikey) (free tier works)

### 2. Clone and install

```bash
git clone <this-repo-url>
cd dad-ai-chatbot-work
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
```

Then edit `.env` and paste your Google API key:

```
GOOGLE_API_KEY=your-actual-key-here
```

## Running

### Demo UI (for testing)

```bash
python ui.py
```

Opens at **http://localhost:7860** — chat interface with text + image upload.

### API server (for web app integration)

```bash
python api.py
```

Runs at **http://localhost:8000** — see API docs at **http://localhost:8000/docs**

### Original CLI (the old way)

```bash
python main.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send message + optional base64 image → full response |
| `POST` | `/chat/stream` | Same but SSE streaming |
| `POST` | `/session/reset` | Reset a session for a new conversation |

Request body for `/chat`:

```json
{
  "message": "My Outlook won't open",
  "image": null,
  "session_id": "my-session"
}
```

Response includes `session_ended: true` when the conversation is finished.

## Project files

| File | Purpose |
|------|---------|
| `agent_core.py` | Shared agent setup, tools, and LLM config |
| `api.py` | FastAPI REST API server |
| `ui.py` | Gradio chat UI (text + image upload) |
| `main.py` | Original CLI chatbot (unchanged) |
| `prompt.txt` | System prompt defining the agent's behavior |
