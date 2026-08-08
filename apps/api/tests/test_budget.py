from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import ProviderBudgetDay, ProviderBudgetReservation
from app.services.budget import (
    BudgetDeniedError,
    BudgetManager,
    estimate_reservation,
)


@pytest.fixture
def budget_manager(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'budget.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield BudgetManager(
        factory,
        daily_limit_usd=Decimal("0.010000"),
        run_limit_usd=Decimal("0.006000"),
        max_attempts_per_run=3,
    )
    engine.dispose()


def reserve(manager: BudgetManager, run_id: str = "run-1", **overrides):
    values = {
        "run_id": run_id,
        "requested_model": "deepseek-flash",
        "input_bytes": 1_000,
        "max_output_tokens": 4_096,
    }
    values.update(overrides)
    return manager.reserve(**values)


def test_reservation_uses_peak_price_and_utf8_safety_bound() -> None:
    amount, estimated_tokens = estimate_reservation(
        requested_model="deepseek-flash",
        input_bytes=1_000,
        max_output_tokens=4_096,
    )

    assert estimated_tokens == 2_000
    assert amount == Decimal("0.005516")


def test_image_reservation_accepts_provider_documented_token_estimate() -> None:
    amount, estimated_tokens = estimate_reservation(
        requested_model="deepseek-flash",
        input_bytes=2_000_000,
        max_output_tokens=512,
        estimated_input_tokens=1_100,
    )

    assert estimated_tokens == 1_100
    assert amount == Decimal("0.000945")


def test_settlement_reconciles_estimate_before_next_reservation(budget_manager) -> None:
    first = reserve(budget_manager)
    with pytest.raises(BudgetDeniedError, match="per-run budget"):
        reserve(budget_manager)

    settled = budget_manager.settle(
        first.reservation_id,
        prompt_tokens=100,
        completion_tokens=100,
    )

    assert settled == Decimal("0.000150")
    second = reserve(budget_manager)
    assert second.reserved_usd == first.reserved_usd


def test_unknown_usage_keeps_full_reservation_charged(budget_manager, tmp_path) -> None:
    first = reserve(budget_manager)
    budget_manager.mark_unknown(first.reservation_id)

    with pytest.raises(BudgetDeniedError, match="per-run budget"):
        reserve(budget_manager)

    engine = create_engine(f"sqlite:///{tmp_path / 'budget.sqlite'}")
    with engine.connect() as connection:
        status = connection.execute(
            select(ProviderBudgetReservation.status).where(
                ProviderBudgetReservation.id == first.reservation_id
            )
        ).scalar_one()
        committed = connection.execute(select(ProviderBudgetDay.committed_usd)).scalar_one()
    engine.dispose()

    assert status == "UNKNOWN"
    assert Decimal(committed) == first.reserved_usd


def test_utc_day_rollover_starts_a_new_daily_allowance(budget_manager) -> None:
    first = reserve(
        budget_manager,
        run_id="first-day-run",
        now=datetime(2026, 10, 5, 23, 59, tzinfo=UTC),
    )
    next_day = reserve(
        budget_manager,
        run_id="next-day-run",
        now=datetime(2026, 10, 6, 0, 1, tzinfo=UTC),
    )

    assert next_day.utc_date > first.utc_date


def test_run_limit_and_attempt_count_continue_across_utc_days(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'budget-rollover.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    manager = BudgetManager(
        factory,
        daily_limit_usd=Decimal("0.010000"),
        run_limit_usd=Decimal("0.010000"),
        max_attempts_per_run=3,
    )

    first = reserve(
        manager,
        run_id="overnight-run",
        now=datetime(2026, 10, 5, 23, 59, tzinfo=UTC),
    )
    manager.settle(
        first.reservation_id,
        prompt_tokens=100,
        completion_tokens=100,
        now=datetime(2026, 10, 5, 23, 59, tzinfo=UTC),
    )
    second = reserve(
        manager,
        run_id="overnight-run",
        now=datetime(2026, 10, 6, 0, 1, tzinfo=UTC),
    )

    # The daily allowance reset, but both attempts still count against the same run.
    assert second.utc_date > first.utc_date
    with pytest.raises(BudgetDeniedError, match="per-run budget"):
        reserve(
            manager,
            run_id="overnight-run",
            now=datetime(2026, 10, 6, 0, 2, tzinfo=UTC),
        )
    engine.dispose()


def test_unknown_model_and_oversized_input_are_refused() -> None:
    with pytest.raises(BudgetDeniedError, match="No current price snapshot"):
        estimate_reservation(
            requested_model="unknown-model",
            input_bytes=100,
            max_output_tokens=1,
        )
    with pytest.raises(BudgetDeniedError, match="paid-input budget"):
        estimate_reservation(
            requested_model="deepseek-flash",
            input_bytes=100_001,
            max_output_tokens=1,
        )
