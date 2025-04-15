import asyncio
import json
from fastapi import FastAPI, BackgroundTasks, WebSocket
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from backend.utils.classes import MainState,ChatState, data_queue, QuestionRequest
from backend.agents.main.agent import build_main_graph

#app = FastAPI()

#@app.post("/process")
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

    final_state = await graph.ainvoke(initial_state)

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

#@app.websocket("/ws/results")
async def stream_results(user_updates: WebSocket):
    await user_updates.accept()  # Accept the WebSocket connection
    try:
        while True:
            item = await data_queue.get()
            await user_updates.send_json(item)

    except Exception as e:
        await user_updates.close()

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


if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run(app, port=8000)
    import asyncio
    
    asyncio.run(process_question(QuestionRequest(
        user_input="What are the taxonomies of the PwC internal repo?",
        history=""
    )))
