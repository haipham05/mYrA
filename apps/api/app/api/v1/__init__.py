from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.papers import router as papers_router
from app.api.v1.projects import router as projects_router

api_router = APIRouter()
api_router.include_router(projects_router)
api_router.include_router(papers_router)
api_router.include_router(jobs_router)
api_router.include_router(chat_router)
