"""Atomic admission controls for paid DeepSeek requests.

Reservations are conservative estimates, not a promise of the provider's final bill.
The snapshot is pinned to the official DeepSeek pricing checked 2026-10-05; peak
prices are used at all times so off-peak discounts cannot inflate available budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, Decimal
from uuid import UUID, uuid4

from sqlalchemy import case, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from app.db.models import ProviderBudgetDay, ProviderBudgetReservation

PRICING_SNAPSHOT = "deepseek-flash-peak-2026-10-05"
PRICING_SNAPSHOT_DATE = date(2026, 10, 5)
PRICE_PER_MILLION = {
    "deepseek-flash": {
        "input_miss": Decimal("0.30"),
        "input_hit": Decimal("0.006"),
        "output": Decimal("1.20"),
    }
}
DAILY_LIMIT_USD = Decimal("1.00")
RUN_LIMIT_USD = Decimal("0.25")
MAX_ATTEMPTS_PER_RUN = 12
MAX_OUTPUT_TOKENS = 4096
MAX_ESTIMATED_INPUT_TOKENS = 200_000
INPUT_TOKEN_SAFETY_MULTIPLIER = 2
MILLION = Decimal(1_000_000)


class BudgetDeniedError(RuntimeError):
    """Raised before transmission when configured spending limits would be exceeded."""


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: UUID
    run_id: str
    utc_date: date
    requested_model: str
    reserved_usd: Decimal
    pricing_snapshot: str


def _ceil_microdollar(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)


def estimate_reservation(
    *,
    requested_model: str,
    input_bytes: int,
    max_output_tokens: int,
    estimated_input_tokens: int | None = None,
) -> tuple[Decimal, int]:
    age_days = (datetime.now(UTC).date() - PRICING_SNAPSHOT_DATE).days
    if age_days > 30:
        raise BudgetDeniedError("DeepSeek pricing snapshot is older than 30 days")
    prices = PRICE_PER_MILLION.get(requested_model)
    if prices is None:
        raise BudgetDeniedError("No current price snapshot is configured for this model")
    input_tokens = (
        estimated_input_tokens
        if estimated_input_tokens is not None
        else input_bytes * INPUT_TOKEN_SAFETY_MULTIPLIER
    )
    if input_tokens < 0:
        raise ValueError("estimated_input_tokens must not be negative")
    if input_tokens > MAX_ESTIMATED_INPUT_TOKENS:
        raise BudgetDeniedError("Request exceeds the configured paid-input budget")
    estimate = (
        Decimal(input_tokens) * prices["input_miss"] + Decimal(max_output_tokens) * prices["output"]
    ) / MILLION
    return _ceil_microdollar(estimate), input_tokens


class BudgetManager:
    """Reserve daily and per-run allowances in the same transaction as each attempt."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        daily_limit_usd: Decimal = DAILY_LIMIT_USD,
        run_limit_usd: Decimal = RUN_LIMIT_USD,
        max_attempts_per_run: int = MAX_ATTEMPTS_PER_RUN,
    ) -> None:
        self._session_factory = session_factory
        self._daily_limit_usd = daily_limit_usd
        self._run_limit_usd = run_limit_usd
        self._max_attempts_per_run = max_attempts_per_run

    def reserve(
        self,
        *,
        run_id: str,
        requested_model: str,
        input_bytes: int,
        max_output_tokens: int,
        estimated_input_tokens: int | None = None,
        now: datetime | None = None,
    ) -> BudgetReservation:
        if not run_id or len(run_id) > 64:
            raise ValueError("run_id must contain 1 to 64 characters")
        if input_bytes < 0 or max_output_tokens < 1:
            raise ValueError("input size and output-token limit are invalid")
        reserved, _ = estimate_reservation(
            requested_model=requested_model,
            input_bytes=input_bytes,
            max_output_tokens=max_output_tokens,
            estimated_input_tokens=estimated_input_tokens,
        )
        utc_date = (now or datetime.now(UTC)).astimezone(UTC).date()
        reservation_id = uuid4()

        with self._session_factory() as db, db.begin():
            dialect = db.get_bind().dialect.name
            if dialect == "postgresql":
                # Serialize one run across UTC-day boundaries as well as within a day.
                db.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:run_id, 0))"),
                    {"run_id": run_id},
                )
            insert = pg_insert if dialect == "postgresql" else sqlite_insert
            db.execute(
                insert(ProviderBudgetDay)
                .values(utc_date=utc_date, committed_usd=Decimal("0"))
                .on_conflict_do_nothing(index_elements=["utc_date"])
            )
            day_budget = db.execute(
                select(ProviderBudgetDay)
                .where(ProviderBudgetDay.utc_date == utc_date)
                .with_for_update()
            ).scalar_one()

            committed_for_run = db.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    ProviderBudgetReservation.status == "SETTLED",
                                    ProviderBudgetReservation.settled_estimate_usd,
                                ),
                                else_=ProviderBudgetReservation.reserved_usd,
                            )
                        ),
                        0,
                    ),
                    func.count(ProviderBudgetReservation.id),
                ).where(
                    ProviderBudgetReservation.run_id == run_id,
                )
            ).one()
            run_spend = Decimal(committed_for_run[0] or 0)
            attempt_count = int(committed_for_run[1] or 0)

            if attempt_count >= self._max_attempts_per_run:
                raise BudgetDeniedError("Run has reached its paid-attempt limit")
            if run_spend + reserved > self._run_limit_usd:
                raise BudgetDeniedError("Request exceeds the remaining per-run budget")
            if Decimal(day_budget.committed_usd or 0) + reserved > self._daily_limit_usd:
                raise BudgetDeniedError("Request exceeds the remaining UTC-day budget")

            day_budget.committed_usd = Decimal(day_budget.committed_usd or 0) + reserved
            db.add(
                ProviderBudgetReservation(
                    id=reservation_id,
                    run_id=run_id,
                    utc_date=utc_date,
                    status="RESERVED",
                    requested_model=requested_model,
                    pricing_snapshot=PRICING_SNAPSHOT,
                    input_bytes=input_bytes,
                    max_output_tokens=max_output_tokens,
                    reserved_usd=reserved,
                )
            )

        return BudgetReservation(
            reservation_id=reservation_id,
            run_id=run_id,
            utc_date=utc_date,
            requested_model=requested_model,
            reserved_usd=reserved,
            pricing_snapshot=PRICING_SNAPSHOT,
        )

    def settle(
        self,
        reservation_id: UUID,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cache_hit_tokens: int | None = None,
        cache_miss_tokens: int | None = None,
        now: datetime | None = None,
    ) -> Decimal | None:
        if prompt_tokens is None or completion_tokens is None:
            self.mark_unknown(reservation_id, now=now)
            return None
        prices = PRICE_PER_MILLION.get("deepseek-flash")
        assert prices is not None
        hit_tokens = max(0, min(prompt_tokens, cache_hit_tokens or 0))
        miss_tokens = (
            max(0, min(prompt_tokens - hit_tokens, cache_miss_tokens))
            if cache_miss_tokens is not None
            else prompt_tokens - hit_tokens
        )
        miss_tokens += prompt_tokens - hit_tokens - miss_tokens
        actual_estimate = _ceil_microdollar(
            (
                Decimal(hit_tokens) * prices["input_hit"]
                + Decimal(miss_tokens) * prices["input_miss"]
                + Decimal(completion_tokens) * prices["output"]
            )
            / MILLION
        )
        timestamp = now or datetime.now(UTC)

        with self._session_factory() as db, db.begin():
            reservation = db.execute(
                select(ProviderBudgetReservation)
                .where(ProviderBudgetReservation.id == reservation_id)
                .with_for_update()
            ).scalar_one_or_none()
            if reservation is None or reservation.status == "SETTLED":
                return actual_estimate if reservation is not None else None
            day_budget = db.execute(
                select(ProviderBudgetDay)
                .where(ProviderBudgetDay.utc_date == reservation.utc_date)
                .with_for_update()
            ).scalar_one()
            day_budget.committed_usd = (
                Decimal(day_budget.committed_usd)
                - Decimal(reservation.reserved_usd)
                + actual_estimate
            )
            reservation.status = "SETTLED"
            reservation.settled_estimate_usd = actual_estimate
            reservation.prompt_tokens = prompt_tokens
            reservation.completion_tokens = completion_tokens
            reservation.settled_at = timestamp
        return actual_estimate

    def mark_unknown(self, reservation_id: UUID, *, now: datetime | None = None) -> None:
        """Keep the full reservation charged when transmission billing is ambiguous."""
        with self._session_factory() as db, db.begin():
            reservation = db.execute(
                select(ProviderBudgetReservation)
                .where(ProviderBudgetReservation.id == reservation_id)
                .with_for_update()
            ).scalar_one_or_none()
            if reservation is not None and reservation.status == "RESERVED":
                reservation.status = "UNKNOWN"
                reservation.settled_at = now or datetime.now(UTC)


_budget_manager: BudgetManager | None = None


def get_budget_manager() -> BudgetManager | None:
    """Enable enforcement only for an explicitly selected local/cloud data profile."""
    global _budget_manager
    from app.config import Settings

    settings = Settings.from_environment()
    if settings.runtime_profile not in {"local", "cloud-data"}:
        return None
    if _budget_manager is None:
        from app.db.session import get_budget_session_factory

        _budget_manager = BudgetManager(get_budget_session_factory(settings))
    return _budget_manager


def set_budget_manager(manager: BudgetManager | None) -> None:
    """Replace the process budget manager for tests or explicit lifecycle setup."""
    global _budget_manager
    _budget_manager = manager
