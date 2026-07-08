import sys
from random import randrange as rand
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from ddgs import DDGS
import langchain
langchain.debug = True
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# LangGraph imports for state and memory
from langgraph.checkpoint.memory import MemorySaver
#load the prompt from prompt.txt
with open("prompt.txt", "r") as f:
    prompt = f.read()

#print(prompt)

@tool
def web_search(query: str) -> str:
    """
    Search the web for information using DDGS.
    
    Args:
        query (str): The search query.
        
    Returns:
        str: The search results as a string.
    """
    # The new package allows you to call it directly without the 'with' block
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
    #write fake ticket file
    ticket_id = rand(1000, 9999)
    with open(f"ticket_{ticket_id}.txt", "w") as f:
        f.write(f"Ticket ID: {ticket_id}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Priority: {priority}\n")
    print(f"Ticket created successfully with ID: {ticket_id}, under category: {category}, assigned to agent: {agent}")

@tool
def end_session():
    """
    End the current session and exit the program.
    """
    print("glad i could help")
    sys.exit(0)




memory = MemorySaver()



llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

agent_executor = create_agent(
    model=llm,
    tools=[create_ticket, web_search, end_session],
    system_prompt=prompt,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "single_session"}}
print("🤖 Agent initialized. I can help you create a ticket.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
        
    print("Agent: ", end="\n", flush=True)
    
    for chunk, _ in agent_executor.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="messages"
    ):
        # 1. Print standard text
        if chunk.content:
            if isinstance(chunk.content, str):
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and "text" in part:
                        print(part["text"], end="", flush=True)
                    elif isinstance(part, str):
                        print(part, end="", flush=True)
        
        # 2. Catch full tool calls
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                print(f"\n[Agent thinking: Decided to use '{tool_call['name']}']... ", end="\n", flush=True)
                
        # 3. Catch tool call chunks
        elif hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            for tool_call in chunk.tool_call_chunks:
                if "name" in tool_call and tool_call["name"]:
                    print(f"\n[Agent thinking: Decided to use '{tool_call['name']}']... ", end="\n", flush=True)

    # Add this empty print statement here! 
    # This ensures the next "You: " starts on a clean line after the stream finishes.
    print()