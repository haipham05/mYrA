"""Read-only API for inspecting the active provider spending allowance."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import ProviderBudgetDay, ProviderBudgetReservation
from app.db.session import get_budget_db
from app.services.budget import DAILY_LIMIT_USD, get_budget_manager

router = APIRouter(prefix="/budget", tags=["budget"])


class BudgetTotals(BaseModel):
    """Aggregated estimates and provider-reported usage for a time scope."""

    committed_estimate_usd: Decimal = Decimal("0")
    settled_estimate_usd: Decimal = Decimal("0")
    active_reservation_usd: Decimal = Decimal("0")
    unknown_reservation_usd: Decimal = Decimal("0")
    reservation_count: int = 0
    reported_prompt_tokens: int | None = None
    reported_completion_tokens: int | None = None


class BudgetUsageResponse(BaseModel):
    """Daily allowance and optional run usage, without provider credentials."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"]
    runtime_profile: str
    unavailable_reason: str | None = None
    utc_date: date
    currency: Literal["USD"] = "USD"
    daily_limit_estimate_usd: Decimal = DAILY_LIMIT_USD
    daily_remaining_estimate_usd: Decimal | None = None
    daily: BudgetTotals
    run_id: str | None = None
    run: BudgetTotals | None = None
    usage_note: str = (
        "Costs are estimates, not provider invoices. Token counts are included only when "
        "reported by the provider; unknown reservations remain charged at their reserved estimate."
    )


def _totals(rows: list[ProviderBudgetReservation], committed: Decimal) -> BudgetTotals:
    settled = sum((row.settled_estimate_usd or Decimal("0") for row in rows), start=Decimal("0"))
    active = sum((row.reserved_usd for row in rows if row.status == "RESERVED"), start=Decimal("0"))
    unknown = sum((row.reserved_usd for row in rows if row.status == "UNKNOWN"), start=Decimal("0"))
    prompt_values = [row.prompt_tokens for row in rows if row.prompt_tokens is not None]
    completion_values = [row.completion_tokens for row in rows if row.completion_tokens is not None]
    return BudgetTotals(
        committed_estimate_usd=committed,
        settled_estimate_usd=settled,
        active_reservation_usd=active,
        unknown_reservation_usd=unknown,
        reservation_count=len(rows),
        reported_prompt_tokens=sum(prompt_values) if prompt_values else None,
        reported_completion_tokens=sum(completion_values) if completion_values else None,
    )


@router.get("/usage", response_model=BudgetUsageResponse)
def get_budget_usage(
    run_id: str | None = Query(default=None, min_length=1, max_length=64),
    db: Session = Depends(get_budget_db),
) -> BudgetUsageResponse:
    """Return today's budget and, when requested, estimates for one run."""
    settings = Settings.from_environment()
    manager = get_budget_manager()
    today = datetime.now(UTC).date()
    if manager is None or settings.runtime_profile not in {"local", "cloud-data"}:
        return BudgetUsageResponse(
            status="unavailable",
            runtime_profile=settings.runtime_profile,
            unavailable_reason=(
                "Budget tracking requires an explicit MYRA_RUNTIME_PROFILE of local or cloud-data."
            ),
            utc_date=today,
            daily_remaining_estimate_usd=None,
            daily=BudgetTotals(),
            run_id=run_id,
            run=BudgetTotals() if run_id else None,
        )

    day_row = db.get(ProviderBudgetDay, today)
    daily_rows = list(
        db.scalars(
            select(ProviderBudgetReservation).where(ProviderBudgetReservation.utc_date == today)
        ).all()
    )
    committed = Decimal(day_row.committed_usd or 0) if day_row is not None else Decimal("0")
    daily = _totals(daily_rows, committed)
    remaining = max(Decimal("0"), DAILY_LIMIT_USD - committed)

    run_totals = None
    if run_id is not None:
        run_rows = list(
            db.scalars(
                select(ProviderBudgetReservation).where(ProviderBudgetReservation.run_id == run_id)
            ).all()
        )
        run_committed = sum(
            (
                row.settled_estimate_usd
                if row.status == "SETTLED" and row.settled_estimate_usd is not None
                else row.reserved_usd
                for row in run_rows
            ),
            start=Decimal("0"),
        )
        run_totals = _totals(run_rows, run_committed)

    return BudgetUsageResponse(
        status="available",
        runtime_profile=settings.runtime_profile,
        utc_date=today,
        daily_remaining_estimate_usd=remaining,
        daily=daily,
        run_id=run_id,
        run=run_totals,
    )
