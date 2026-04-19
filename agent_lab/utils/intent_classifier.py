from langchain_core.messages import SystemMessage, HumanMessage
import json


def classify_intent(user_msg: str, invoke_llm_fn) -> str:
    try:
        messages = [
            SystemMessage(content="Classify the user's message into one of these intents: math, search, or chat. Respond in valid JSON only with format: {\"intent\": \"...\"}"),
            HumanMessage(content=user_msg)
        ]
        response = invoke_llm_fn(messages)
        data = json.loads(response.content)
        intent = data["intent"].lower().strip()
        if intent in {"math", "search", "chat"}:
            return intent
        return "chat"
    except Exception:
        return "chat"




