from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1 import budget as budget_api
from app.db.base import Base
from app.db.models import ProviderBudgetDay
from app.db.session import get_db
from app.main import app
from app.services.budget import BudgetManager


@pytest.fixture
def budget_api_context(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'budget-api.sqlite'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    manager = BudgetManager(factory)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setenv("MYRA_RUNTIME_PROFILE", "local")
    monkeypatch.setattr(budget_api, "get_budget_manager", lambda: manager)
    with TestClient(app) as client:
        yield client, session, manager
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_budget_usage_requires_explicit_profile(budget_api_context, monkeypatch):
    client, _, _ = budget_api_context
    monkeypatch.setenv("MYRA_RUNTIME_PROFILE", "auto")
    monkeypatch.setattr(budget_api, "get_budget_manager", lambda: None)

    response = client.get("/api/v1/budget/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["runtime_profile"] == "auto"
    assert "explicit MYRA_RUNTIME_PROFILE" in body["unavailable_reason"]
    assert body["daily_remaining_estimate_usd"] is None


def test_budget_usage_active_profile_reports_available(budget_api_context):
    client, _, _ = budget_api_context

    response = client.get("/api/v1/budget/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["runtime_profile"] == "local"
    assert body["currency"] == "USD"
    assert body["daily_limit_estimate_usd"] == "1.00"
    assert body["daily_remaining_estimate_usd"] == "1.00"
    assert "not provider invoices" in body["usage_note"]
    assert not any("key" in name.lower() for name in body)


def test_budget_usage_empty_day_returns_zero_totals(budget_api_context):
    client, session, _ = budget_api_context
    today = datetime.now(UTC).date()
    session.add(ProviderBudgetDay(utc_date=today, committed_usd=Decimal("0")))
    session.commit()

    response = client.get("/api/v1/budget/usage")

    assert response.status_code == 200
    daily = response.json()["daily"]
    assert daily["committed_estimate_usd"] == "0"
    assert daily["settled_estimate_usd"] == "0"
    assert daily["reservation_count"] == 0
    assert daily["reported_prompt_tokens"] is None
    assert daily["reported_completion_tokens"] is None


def test_budget_usage_includes_run_estimates_and_reported_tokens(budget_api_context):
    client, _, manager = budget_api_context
    reservation = manager.reserve(
        run_id="research-run-1",
        requested_model="deepseek-flash",
        input_bytes=100,
        max_output_tokens=100,
    )
    manager.settle(
        reservation.reservation_id,
        prompt_tokens=31,
        completion_tokens=17,
    )
    unknown = manager.reserve(
        run_id="research-run-2",
        requested_model="deepseek-flash",
        input_bytes=80,
        max_output_tokens=100,
    )
    manager.mark_unknown(unknown.reservation_id)

    response = client.get("/api/v1/budget/usage?run_id=research-run-1")

    assert response.status_code == 200
    body = response.json()
    assert body["daily"]["reservation_count"] == 2
    assert body["daily"]["reported_prompt_tokens"] == 31
    assert body["daily"]["reported_completion_tokens"] == 17
    assert Decimal(body["daily"]["unknown_reservation_usd"]) > 0
    assert body["run_id"] == "research-run-1"
    assert body["run"]["reservation_count"] == 1
    assert body["run"]["reported_prompt_tokens"] == 31
    assert body["run"]["reported_completion_tokens"] == 17
    assert Decimal(body["run"]["settled_estimate_usd"]) > 0


def test_budget_usage_run_totals_include_prior_utc_days(budget_api_context):
    client, _, manager = budget_api_context
    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)
    previous = manager.reserve(
        run_id="overnight-run",
        requested_model="deepseek-flash",
        input_bytes=100,
        max_output_tokens=100,
        now=yesterday,
    )
    manager.settle(
        previous.reservation_id,
        prompt_tokens=20,
        completion_tokens=10,
        now=yesterday,
    )
    manager.reserve(
        run_id="overnight-run",
        requested_model="deepseek-flash",
        input_bytes=80,
        max_output_tokens=100,
        now=today,
    )

    response = client.get("/api/v1/budget/usage?run_id=overnight-run")

    assert response.status_code == 200
    body = response.json()
    assert body["daily"]["reservation_count"] == 1
    assert body["run"]["reservation_count"] == 2
    assert body["run"]["reported_prompt_tokens"] == 20
