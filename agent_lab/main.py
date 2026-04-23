from agent.graph import build_agent

THREAD_ID = "session-1"

def main():
    agent = build_agent()
    print("Personal AI Assistant v1.0 | Type 'exit' to quit\n")

    while True:
        q = input("You: ").strip()
        if not q:
            print("(empty input, try again)\n")
            continue
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