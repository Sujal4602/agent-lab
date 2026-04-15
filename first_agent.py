import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def calculator(expression: str) -> str:
    """Evaluate math expressions like 25*4+10"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    print("Agent is thinking...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

memory = MemorySaver()

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session_1"}}

print("Agent ready! Type 'exit' to quit.")
while True:
    user_input = input("How may i help you: ")
    if user_input.lower() == "exit":
        break
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print("Agent:", result["messages"][-1].content)
    print()
    