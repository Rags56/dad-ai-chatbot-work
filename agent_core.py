import uuid
import os
from random import randrange as rand
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from ddgs import DDGS
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

if "GOOGLE_API_KEY" not in os.environ:
    import getpass
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

@tool
def web_search(query: str) -> str:
    """
    Search the web for information using DDGS.
    
    Args:
        query (str): The search query.
        
    Returns:
        str: The search results as a string.
    """
    results = DDGS().text(query, max_results=5)
    return str(results)

@tool
def create_ticket(description: str, category: str, agent: str, priority: str) -> str:
    """
    Create a ticket in the ticketing system.

    Args:
        description (str): Description of the issue.
        category (str): Category of the issue.
        agent (str): Agent assigned to the ticket.
        priority (str): Priority of the ticket.

    Returns:
        str: Confirmation message with ticket ID.
    """
    ticket_id = rand(1000, 9999)
    with open(f"ticket_{ticket_id}.txt", "w") as f:
        f.write(f"Ticket ID: {ticket_id}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Priority: {priority}\n")
    print(f"Ticket created successfully with ID: {ticket_id}, under category: {category}, assigned to agent: {agent}")

@tool
def end_session() -> str:
    """
    End the current support session. Call this after either:
    A) The user's problem has been solved, or
    B) A support ticket has been created.
    Always call this to properly close the conversation.
    """
    return "Session ended. Thank you for using Abu Odeh IT Support."

ended_sessions: set = set()
_session_threads: dict[str, str] = {}


def get_thread_id(session_id: str) -> str:
    if session_id not in _session_threads:
        _session_threads[session_id] = uuid.uuid4().hex
    return _session_threads[session_id]


def reset_thread(session_id: str) -> str:
    new_id = uuid.uuid4().hex
    _session_threads[session_id] = new_id
    return new_id

_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_script_dir, "prompt.txt"), "r") as f:
    prompt = f.read()

memory = MemorySaver()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

agent_executor = create_agent(
    model=llm,
    tools=[create_ticket, web_search, end_session],
    system_prompt=prompt,
    checkpointer=memory
)
