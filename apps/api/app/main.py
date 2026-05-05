from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="mYrA API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=[],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok", "service": "myra-api"}
