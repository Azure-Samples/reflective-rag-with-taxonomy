from email import message
import httpx
from chat_memory import ChatMemory

FASTAPI_ENDPOINT = "http://127.0.0.1:8000/process"  # Adjust if using different port

def simulate_chat(session_id: str, user_input: str, memory: ChatMemory):
    memory.add_message(session_id, sender="user", role="user", content=user_input)

    history_str = "\n".join([f"{msg['role']} ({msg['sender']}): {msg['content']}" for msg in memory.get_history(session_id)])
    print("History String :" + history_str)
    # Call FastAPI endpoint
    try:
        response = httpx.post(FASTAPI_ENDPOINT, json={"user_input": user_input,"history":history_str},
    timeout=60.0)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("final_answer", "No answer generated.")
        else:
            answer = f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        answer = f"Request failed: {e}"

    memory.add_message(
        session_id,
        sender="multi-agent-rag",
        role="agent",
        content=answer
    )

    return {"final_answer": answer}

if __name__ == "__main__":
    memory = ChatMemory()
    session_id = "session_1" #use UUID if need to be saved in DB 

    print("Multi-Agent RAG CLI (via FastAPI) with Chat Memory")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = simulate_chat(session_id, user_input, memory)

        print("\nAssistant:", result.get("final_answer"))
        print("\n--- Chat History ---")
        for msg in memory.get_history(session_id):
            print(f"[{msg['timestamp']}] {msg['role']} ({msg['sender']}): {msg['content']}")
