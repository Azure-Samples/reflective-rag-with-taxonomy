import asyncio
import json
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from backend.utils.classes import MainState,ChatState
from backend.agents.main.agent import build_main_graph,graph_invoke

app = FastAPI()

class QuestionRequest(BaseModel):
    user_input: str
    history: str

async def graph_invoke_queue(state: dict):
    """Simulating graph processing - yields events in an async manner."""
    await asyncio.sleep(60)  # Simulate async work
    yield {"event_type": "start", "message": "Processing started"}
    await asyncio.sleep(60)
    yield {"event_type": "info", "message": "Processing some data"}
    await asyncio.sleep(60)
    yield {"event_type": "end", "message": "Processing finished"}

async def stream_chat_response(user_input: str, event_queue: asyncio.Queue):
    """Async generator that runs the agent graph in the background and yields SSE events."""
    initial_state = {
        "user_input": user_input,
        "current_results": [],
        "vetted_results": [],
        "discarded_results": [],
        "processed_ids": set(),
        "reviews": [],
        "decisions": [],
        "final_answer": None,
        "attempts": 0,
        "search_history": [],
        "thought_process": []
    }

    # Run the graph_invoke in the background and collect events.
    async def run_graph():
        try:
            async for event in graph_invoke(initial_state):
                await event_queue.put(event)
        except Exception as e:
            await event_queue.put({"event_type": "server-error", "message": str(e)})
        # Always signal completion.
        await event_queue.put({"event_type": "end"})

    # Start the graph processing in the background
    asyncio.create_task(run_graph())

    # Yield SSE events from the queue.
    while True:
        event = await event_queue.get()  # Non-blocking asynchronous get
        sse_message = f"event: {event['event_type']}\n" \
                      f"data: {json.dumps(event)}\n\n"
        yield sse_message
        if event.get("event_type") == "end":
            break

@app.post("/conversation")
async def conversation(request: QuestionRequest, background_tasks: BackgroundTasks):
    """Handles the incoming conversation request and streams events back via SSE."""
    user_input = request.user_input
    history = request.history

    # Use asyncio.Queue to handle events asynchronously
    event_queue = asyncio.Queue()

    # Start graph processing in the background and yield SSE events from event_queue
    background_tasks.add_task(stream_chat_response, user_input, event_queue)

    # Return an SSE response with streamed events
    return EventSourceResponse(stream_chat_response(user_input, event_queue))

@app.post("/process")
async def process_question(request: QuestionRequest):
    user_input = request.user_input
    history = request.history

    graph = build_main_graph()
    initial_state = MainState(
        user_input=user_input,
        user_history=history,
        taxonomies=[],
        research_results=[],
        research_outputs=[],
        final_answer=None,
        thought_process=[],
    )

    final_state = graph.invoke(initial_state)

    if final_state["final_answer"]:

        return JSONResponse(
            {
                "final_answer": final_state["final_answer"],
                "taxonomies": final_state["taxonomies"],
                "research_results": final_state["research_results"],
                "thought_process": final_state["thought_process"],
            }
        )
    else:
        return JSONResponse(
            {"error": "Unable to find a satisfactory answer."}, status_code=400
        )

# from fastapi import FastAPI
# from fastapi.responses import PlainTextResponse
# import sys

# app = FastAPI()


# @app.get("/")
# async def index():
#     version = sys.version_info
#     return PlainTextResponse(
#         content=f"Hello World, I am Python {version.major}.{version.minor}"
#     )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)
