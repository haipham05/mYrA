from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Project
from app.schemas.project import ProjectCreate


def create_project(db: Session, project_in: ProjectCreate) -> Project:
    project = Project(name=project_in.name, description=project_in.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_project(db: Session, project_id: UUID) -> Project | None:
    return db.query(Project).filter(Project.id == project_id).first()


def list_projects(db: Session, limit: int = 50, offset: int = 0) -> tuple[list[Project], int]:
    query = db.query(Project).order_by(Project.created_at.desc())
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return items, total
