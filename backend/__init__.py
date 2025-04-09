from fastapi import FastAPI
from fastapi.responses import JSONResponse
from backend.utils.classes import MainState
from backend.agents.main.agent import build_main_graph

app = FastAPI()

@app.post("/process")
async def process_question(user_input: str):

    graph = build_main_graph()
    initial_state = MainState(
        user_input=user_input,
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
