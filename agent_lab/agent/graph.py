import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

load_dotenv(dotenv_path=r"C:\Users\DELL\Downloads\ai_agents\.env")

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

from tools.calculator import calculator_tool
from tools.search import search_tool
from memory.memory import get_memory


def build_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    tools = [calculator_tool, search_tool]
    llm_with_tools = llm.bind_tools(tools)
    memory = get_memory()

    def call_llm(state):
    system = SystemMessage(content=(
        "You are a helpful conversational assistant with access to tools. "
        "ONLY use calculator_tool when user explicitly asks for a math calculation or percentage. "
        "ONLY use search_tool when user asks about real world facts, news, or current events. "
        "For ALL other questions respond normally in plain text. "
        "Never use XML tags, brackets, or special formatting in responses. "
        "Plain conversational text only."
        "When search_tool returns results, copy the results into your response word for word. Do not summarize, rephrase, or add anything."
    ))
    messages = [system] + state["messages"]
    
    # force tool choice when search intent detected
    last_msg = state["messages"][-1].content.lower()
    search_keywords = ["news", "latest", "current", "who is", "what is", "search", "find"]
    math_keywords = ["percent", "calculate", "%", "+", "-", "*", "/"]
    
    if any(k in last_msg for k in search_keywords):
        return {"messages": [llm_with_tools.invoke(messages, tool_choice={"type": "function", "function": {"name": "search_tool"}})]}
    elif any(k in last_msg for k in math_keywords):
        return {"messages": [llm_with_tools.invoke(messages, tool_choice={"type": "function", "function": {"name": "calculator_tool"}})]}
    
    return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=memory)