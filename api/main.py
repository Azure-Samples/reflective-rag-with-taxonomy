from fastapi import FastAPI
from api.router import router

app = FastAPI(
    title="Research Agent API",
    version="1.0.0"
)

app.include_router(router, prefix="/api")
