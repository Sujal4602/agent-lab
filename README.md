# agent-lab

Building AI agents from scratch. Each project ships a working, tested agent.

---

## Project 1 — Personal AI Assistant Agent

A CLI-based conversational agent with tool calling, web search, and persistent memory across sessions.

**What it does**
- Answers math questions via calculator tool
- Searches the web for real-world facts via Tavily
- Remembers conversation history across sessions (SQLite)
- Trims history to prevent token bloat (last 20 messages sent to LLM)

**Stack**
- Python
- LangGraph — agent state machine
- LangChain — tool binding
- Groq (llama-3.3-70b-versatile) — LLM
- Tavily — web search
- SQLite — persistent memory
- Tenacity — retry on rate limits

**Setup**
```bash
git clone https://github.com/Sujal4602/agent-lab
cd agent-lab
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` file:
```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

Run:
```bash
python main.py
```

**Usage**
```
You: who is Elon Musk
Agent: Elon Musk is...

You: what is 35% of 20000
Agent: 7000.0

You: exit
```
