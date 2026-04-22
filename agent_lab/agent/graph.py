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
            'You are a helpful assistant with access to two tools: calculator_tool and search_tool. '
            'Use calculator_tool for any math, arithmetic, or percentage calculation. '
            'Use search_tool for any question about real people, current events, or world facts. '
            'For greetings, personal questions, or memory recall — respond directly without tools. '
            'When tool results are available, base your answer ONLY on those results. '
            'Plain conversational text only. No XML tags.'
        ))

        messages = [system] + state["messages"]

        last_msg_obj = state["messages"][-1]
        if last_msg_obj.type == "tool":
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        last_msg = last_msg_obj.content
        logger.info(f"User: {last_msg}")

        @groq_retry
        def safe_invoke(msgs):
            return llm_with_tools.invoke(msgs)

        response = safe_invoke(messages)

        log_content = response.content if response.content else "[tool_call]"
        logger.info(f"Agent: {log_content}")
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=memory)