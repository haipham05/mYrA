from fastapi import APIRouter

from app.api.v1.artifacts import router as artifacts_router
from app.api.v1.assistant import router as assistant_router
from app.api.v1.budget import router as budget_router
from app.api.v1.chat import router as chat_router
from app.api.v1.graph import router as graph_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.memory import router as memory_router
from app.api.v1.papers import router as papers_router
from app.api.v1.projects import router as projects_router
from app.api.v1.translations import router as translations_router

api_router = APIRouter()
api_router.include_router(artifacts_router)
api_router.include_router(assistant_router)
api_router.include_router(budget_router)
api_router.include_router(projects_router)
api_router.include_router(papers_router)
api_router.include_router(jobs_router)
api_router.include_router(chat_router)
api_router.include_router(memory_router)
api_router.include_router(graph_router)
api_router.include_router(translations_router)
