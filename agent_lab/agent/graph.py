import os
import sqlite3
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

load_dotenv()

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

from tools.calculator import calculator_tool
from tools.search import search_tool
from utils.Retry import groq_retry
from utils.logger import get_logger
from utils.intent_classifier import classify_intent
from memory.memory import get_memory

logger = get_logger("agent")


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
            "Plain conversational text only. "
            "When search_tool returns results, summarize them clearly in plain text."
        ))

        messages = [system] + state["messages"]

        last_msg_obj = state["messages"][-1]
        if last_msg_obj.type == "tool":
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        last_msg = last_msg_obj.content
        logger.info(f"User: {last_msg}")

        @groq_retry
        def safe_invoke(msgs, **kwargs):
            return llm_with_tools.invoke(msgs, **kwargs)

        intent = classify_intent(last_msg, llm.invoke)
        logger.info(f"Intent: {intent}")

        if intent == "search":
            response = safe_invoke(messages, tool_choice={"type": "function", "function": {"name": "search_tool"}})
        elif intent == "math":
            response = safe_invoke(messages, tool_choice={"type": "function", "function": {"name": "calculator_tool"}})
        else:
            response = safe_invoke(messages)

        logger.info(f"Agent: {response.content}")
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=memory)