import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def get_memory():
    os.makedirs("memory", exist_ok=True)
    conn = sqlite3.connect("memory/agent_memory.db", check_same_thread=False)
    return SqliteSaver(conn)