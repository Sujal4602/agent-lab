
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def call_model(state: MessagesState):
    print("Agent is thinking...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_edge(START, "agent")

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="Who are you?")]})
print(result["messages"][-1].content)

