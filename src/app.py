from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import sys

app = FastAPI()


@app.get("/")
async def index():
    version = sys.version_info
    return PlainTextResponse(
        content=f"Hello World, I am Python {version.major}.{version.minor}"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
