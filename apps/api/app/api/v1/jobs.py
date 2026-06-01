from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.crud.job import get_job
from app.db.session import get_db
from app.schemas.job import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_single_job(
    job_id: UUID,
    db: Session = Depends(get_db),
) -> JobResponse:
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return JobResponse.model_validate(job, from_attributes=True)


@router.post("/{job_id}/retry", response_model=JobResponse)
def retry_single_job(
    job_id: UUID,
    db: Session = Depends(get_db),
) -> JobResponse:
    from app.crud.job import retry_job

    job = get_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    if job.status == "PROCESSING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot retry a job that is currently processing",
        )
    if job.status == "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot retry an already completed job",
        )
    try:
        updated_job = retry_job(db, job_id)
        return JobResponse.model_validate(updated_job, from_attributes=True)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
