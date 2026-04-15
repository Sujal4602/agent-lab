import os
import ast
import operator as op
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# Safe math operators
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg
}

def safe_eval(node):
    if isinstance(node, ast.Constant):  # numbers only
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers allowed")

    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed")

        return ALLOWED_OPERATORS[operator_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Unary operator not allowed")

        return ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Invalid expression")

@tool
def calculator(expression: str) -> str:
    """Use this tool ONLY for pure math expressions containing numbers and operators like +, -, *, /, %, **, and parentheses."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = safe_eval(parsed.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)
def call_model(state: MessagesState):
    print("Agent is thinking...")

    system_prompt = SystemMessage(
        content=(
            "You are a personal multi-tool AI assistant. "
            "Use important facts shared by the user in the current session "
            "to personalize future responses. "
            "Remember previous user messages in the same session and use them "
            "for follow-up questions."
        )
    )

    last_message = state["messages"][-1].content.lower()

    # Only enable tools for clear math or web-search style queries
    if any(op in last_message for op in ["+", "-", "*", "/", "%", "**"]) or "latest" in last_message:
        response = llm_with_tools.invoke(
            [system_prompt] + state["messages"]
        )
    else:
        response = llm.invoke(
            [system_prompt] + state["messages"]
        )

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
