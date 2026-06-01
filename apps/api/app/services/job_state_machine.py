from typing import NamedTuple

from app.schemas.job import JobStage, JobStatus


class InvalidJobTransitionError(ValueError):
    """Raised when an illegal job status or stage transition is attempted."""


class StateTuple(NamedTuple):
    status: JobStatus | str
    stage: JobStage | str


# Formal transition table: mapping current (status, stage) to set of allowable next (status, stage)
LEGAL_TRANSITIONS: dict[StateTuple, set[StateTuple]] = {
    # 1. PENDING / QUEUED: can transition to PROCESSING (claim) or FAILED (reconciled/canceled)
    StateTuple(JobStatus.PENDING, JobStage.QUEUED): {
        StateTuple(JobStatus.PROCESSING, JobStage.PARSING),
        StateTuple(JobStatus.FAILED, JobStage.FAILED),
    },
    # 2. PROCESSING / PARSING
    StateTuple(JobStatus.PROCESSING, JobStage.PARSING): {
        StateTuple(JobStatus.PROCESSING, JobStage.PARSING),  # progress update
        StateTuple(JobStatus.PROCESSING, JobStage.CHUNKING),
        StateTuple(JobStatus.PENDING, JobStage.QUEUED),  # lease lost / released / retry
        StateTuple(JobStatus.FAILED, JobStage.FAILED),  # terminal error
    },
    # 3. PROCESSING / CHUNKING
    StateTuple(JobStatus.PROCESSING, JobStage.CHUNKING): {
        StateTuple(JobStatus.PROCESSING, JobStage.CHUNKING),
        StateTuple(JobStatus.PROCESSING, JobStage.EMBEDDING),
        StateTuple(JobStatus.PENDING, JobStage.QUEUED),
        StateTuple(JobStatus.FAILED, JobStage.FAILED),
    },
    # 4. PROCESSING / EMBEDDING
    StateTuple(JobStatus.PROCESSING, JobStage.EMBEDDING): {
        StateTuple(JobStatus.PROCESSING, JobStage.EMBEDDING),
        StateTuple(JobStatus.PROCESSING, JobStage.INDEXING),
        StateTuple(JobStatus.PENDING, JobStage.QUEUED),
        StateTuple(JobStatus.FAILED, JobStage.FAILED),
    },
    # 5. PROCESSING / INDEXING
    StateTuple(JobStatus.PROCESSING, JobStage.INDEXING): {
        StateTuple(JobStatus.PROCESSING, JobStage.INDEXING),
        StateTuple(JobStatus.COMPLETED, JobStage.COMPLETED),  # successful completion
        StateTuple(JobStatus.PENDING, JobStage.QUEUED),
        StateTuple(JobStatus.FAILED, JobStage.FAILED),
    },
    # 6. COMPLETED / COMPLETED: terminal state (can only be reset via explicit retry)
    StateTuple(JobStatus.COMPLETED, JobStage.COMPLETED): set(),
    # 7. FAILED / FAILED: terminal state, but can be retried to PENDING / QUEUED
    StateTuple(JobStatus.FAILED, JobStage.FAILED): {
        StateTuple(JobStatus.PENDING, JobStage.QUEUED),  # explicit retry
    },
}


def validate_job_transition(
    current_status: JobStatus | str,
    current_stage: JobStage | str,
    next_status: JobStatus | str,
    next_stage: JobStage | str,
) -> None:
    """Validates that a job transition is legal according to the state machine table.

    Raises InvalidJobTransitionError if the transition is prohibited.
    """
    curr = StateTuple(
        current_status.value if isinstance(current_status, JobStatus) else current_status,
        current_stage.value if isinstance(current_stage, JobStage) else current_stage,
    )
    nxt = StateTuple(
        next_status.value if isinstance(next_status, JobStatus) else next_status,
        next_stage.value if isinstance(next_stage, JobStage) else next_stage,
    )

    # Identical state is a no-op progress update
    if curr == nxt:
        return

    allowed = LEGAL_TRANSITIONS.get(curr)
    if allowed is None or nxt not in allowed:
        raise InvalidJobTransitionError(
            f"Illegal job transition from status={curr.status}, stage={curr.stage} "
            f"to status={nxt.status}, stage={nxt.stage}"
        )
