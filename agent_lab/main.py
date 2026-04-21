from agent.graph import build_agent

THREAD_ID = "session-1"

def main():
    agent = build_agent()
    print("Agent Ready (Day 12)\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"configurable": {"thread_id": THREAD_ID}}
        )

        last_message = result["messages"][-1]
        print("Agent:", last_message.content, "\n")

if __name__ == "__main__":
    main()