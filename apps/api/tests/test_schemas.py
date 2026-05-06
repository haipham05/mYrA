from datetime import UTC, datetime
from uuid import uuid4

from app.schemas.chat import (
    ConversationResponse,
    MessageResponse,
    MessageRole,
)
from app.schemas.evidence import (
    BoundingBox,
    Citation,
    CoordinateOrigin,
    EvidenceItem,
    SourceElement,
)
from app.schemas.job import JobResponse, JobStage, JobStatus
from app.schemas.paper import PaperResponse, PaperStatus, PaperUploadResponse
from app.schemas.project import ProjectResponse


def test_bounding_box_normalization_top_left() -> None:
    box = BoundingBox(
        x_min=100,
        y_min=200,
        x_max=300,
        y_max=400,
        page_width=1000,
        page_height=2000,
        origin=CoordinateOrigin.TOP_LEFT,
    )
    norm = box.to_normalized_top_left()
    assert norm.x_min == 0.1
    assert norm.y_min == 0.1
    assert norm.x_max == 0.3
    assert norm.y_max == 0.2
    assert norm.origin == CoordinateOrigin.TOP_LEFT


def test_bounding_box_normalization_bottom_left() -> None:
    # In bottom-left: y_min=200, y_max=400 on page_height=2000
    # In top-left: y_min should be (2000 - 400) / 2000 = 1600 / 2000 = 0.8
    # y_max should be (2000 - 200) / 2000 = 1800 / 2000 = 0.9
    box = BoundingBox(
        x_min=100,
        y_min=200,
        x_max=300,
        y_max=400,
        page_width=1000,
        page_height=2000,
        origin=CoordinateOrigin.BOTTOM_LEFT,
    )
    norm = box.to_normalized_top_left()
    assert norm.x_min == 0.1
    assert norm.y_min == 0.8
    assert norm.x_max == 0.3
    assert norm.y_max == 0.9
    assert norm.origin == CoordinateOrigin.TOP_LEFT


def test_bounding_box_zero_dimensions_graceful() -> None:
    box = BoundingBox(
        x_min=0,
        y_min=0,
        x_max=0,
        y_max=0,
        page_width=0,
        page_height=0,
        origin=CoordinateOrigin.TOP_LEFT,
    )
    norm = box.to_normalized_top_left()
    assert norm == box


def test_schemas_instantiation() -> None:
    now = datetime.now(tz=UTC)
    proj_id = uuid4()
    paper_id = uuid4()
    job_id = uuid4()
    conv_id = uuid4()
    msg_id = uuid4()
    elem_id = uuid4()
    chunk_id = uuid4()

    proj = ProjectResponse(id=proj_id, name="Test Project", created_at=now, updated_at=now)
    assert proj.name == "Test Project"

    paper_upload = PaperUploadResponse(
        paper_id=paper_id, job_id=job_id, status=PaperStatus.PROCESSING
    )
    assert paper_upload.status == PaperStatus.PROCESSING

    paper = PaperResponse(
        id=paper_id,
        project_id=proj_id,
        filename="test.pdf",
        status=PaperStatus.READY,
        page_count=5,
        created_at=now,
        updated_at=now,
    )
    assert paper.status == PaperStatus.READY

    job = JobResponse(
        id=job_id,
        paper_id=paper_id,
        status=JobStatus.COMPLETED,
        stage=JobStage.COMPLETED,
        progress=1.0,
        created_at=now,
        updated_at=now,
    )
    assert job.status == JobStatus.COMPLETED

    elem = SourceElement(
        id=elem_id,
        element_index=0,
        element_type="text",
        text="Sample paragraph",
        page_number=1,
    )
    assert elem.element_type == "text"

    evidence = EvidenceItem(
        id="E1",
        paper_id=paper_id,
        chunk_id=chunk_id,
        quote="Sample paragraph quote",
        page_number=1,
    )
    citation = Citation(
        citation_index=1,
        evidence_id="E1",
        paper_id=paper_id,
        page_number=1,
        quote="Sample paragraph quote",
    )

    conv = ConversationResponse(id=conv_id, project_id=proj_id, created_at=now, updated_at=now)
    assert conv.id == conv_id

    msg = MessageResponse(
        id=msg_id,
        conversation_id=conv_id,
        role=MessageRole.ASSISTANT,
        content="Answer [1]",
        citations=[citation],
        evidence=[evidence],
        created_at=now,
    )
    assert msg.role == MessageRole.ASSISTANT
    assert len(msg.citations) == 1
