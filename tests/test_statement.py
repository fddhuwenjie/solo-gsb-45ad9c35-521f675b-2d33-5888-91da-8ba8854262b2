"""Tests for consolidated multi-cycle settlement & carbon statements.

Like the settlement/closing/carbon tests, these run against a temporary
*file-backed* SQLite database (WAL, real write locks) so concurrent
statement creation exercises the actual serialization guarantees.
"""
from __future__ import annotations

import os
import threading
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402

from app import carbon, closing, settlement, statement  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    BillingCycle,
    CarbonReport,
    CycleClosure,
    RateRule,
    Statement,
    TariffVersion,
    Tier,
    configure_sqlite_engine,
    seed,
)
from app.main import app  # noqa: E402
from app.pricing import PricingError  # noqa: E402
from app.schemas import (  # noqa: E402
    CarbonReportRequest,
    CycleCloseRequest,
    QuoteIntervalIn,
    SettlementBatchRequest,
    StatementRequest,
)

V = "us-ny-demo-v2024"
CN = "cn-shanghai-demo-v2024"
# 2024-07-06/07 are a weekend in NY -> only the `leisure` rule
# (tiers: 100 @0.15, 300 @0.18, open @0.22).
A1 = {
    "start": "2024-07-06T00:00:00-04:00",
    "end": "2024-07-06T01:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}  # 04:00-05:00 UTC, 60 kWh -> 60 x 0.15 = 9.00
B1 = {
    "start": "2024-07-07T00:00:00-04:00",
    "end": "2024-07-07T01:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}  # 2024-07-07 04:00-05:00 UTC
B2 = {
    "start": "2024-07-07T01:00:00-04:00",
    "end": "2024-07-07T02:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}  # 2024-07-07 05:00-06:00 UTC; A+B2 style tier split -> total 18.60
C1 = {
    "start": "2024-07-06T00:00:00+08:00",
    "end": "2024-07-06T01:00:00+08:00",
    "timezone": "Asia/Shanghai",
    "energy": "60",
}  # 2024-07-05 16:00-17:00 UTC, Shanghai weekend -> 60 x 0.617 = 37.02

FA = [{"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
       "factor": "0.4"}]
FB = [{"start": "2024-07-07T04:00:00Z", "end": "2024-07-07T06:00:00Z",
       "factor": "0.5"}]
FC = [{"start": "2024-07-05T16:00:00Z", "end": "2024-07-05T17:00:00Z",
       "factor": "0.6"}]


@pytest.fixture()
def tmp_engine(tmp_path):
    path = tmp_path / "statement.db"
    eng = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng)
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        seed(s)
        s.commit()
    yield eng, path
    eng.dispose()


def _submit(eng, cycle, key, rev, intervals, version=V):
    req = SettlementBatchRequest(
        version=version, idempotency_key=key, expected_revision=rev,
        intervals=[QuoteIntervalIn(**iv) for iv in intervals])
    return settlement.submit_batch(cycle, req, engine=eng)


def _close(eng, cycle, close_id, rev, total):
    req = CycleCloseRequest(close_id=close_id, expected_revision=rev,
                            expected_total_amount=Decimal(total))
    return closing.close_cycle(cycle, req, engine=eng)


def _gen(eng, cycle, rev, factors, factor_version="grid-2024-v1"):
    req = CarbonReportRequest(
        factor_version=factor_version, expected_revision=rev,
        factors=[dict(f) for f in factors])
    return carbon.generate_report(cycle, req, engine=eng)


def _create(eng, statement_id, cycles):
    req = StatementRequest(statement_id=statement_id, cycles=cycles)
    return statement.create_statement(req, engine=eng)


def _cycle_entry(eng, cycle, batches, close_id, total, factors, version=V):
    """Submit batches, close and report one cycle; return its statement
    request entry (cycle_id + both expected digests)."""
    for key, rev, ivs in batches:
        _submit(eng, cycle, key, rev, ivs, version=version)
    rev = len(batches)
    closed = _close(eng, cycle, close_id, rev, total).body
    report = _gen(eng, cycle, rev, factors).body
    return {"cycle_id": cycle,
            "detail_digest": closed["detail_digest"],
            "report_digest": report["report_digest"]}


def _cycle_a(eng, cycle="cy-a"):
    return _cycle_entry(eng, cycle, [("k1", 0, [A1])], f"close-{cycle}",
                        "9.00", FA)


def _cycle_b(eng, cycle="cy-b"):
    return _cycle_entry(eng, cycle, [("k1", 0, [B1]), ("k2", 1, [B2])],
                        f"close-{cycle}", "18.60", FB)


def _cycle_c(eng, cycle="cy-c"):
    return _cycle_entry(eng, cycle, [("k1", 0, [C1])], f"close-{cycle}",
                        "37.02", FC, version=CN)


def _count(eng, model):
    with Session(eng) as s:
        return len(s.scalars(select(model)).all())


# ------------------------------------------------------------- happy paths


def test_create_statement_two_cycles(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    res = _create(eng, "st-1", [a, b])
    assert res.replayed is False
    body = res.body
    assert body["statement_id"] == "st-1"
    assert body["cycle_count"] == 2
    assert body["currency"] == "USD"
    assert body["energy_unit"] == "kWh"
    assert body["emission_unit"] == "kgCO2e"
    assert [c["cycle_id"] for c in body["cycles"]] == ["cy-a", "cy-b"]

    ca, cb = body["cycles"]
    # per-cycle values come straight from the closure/report rows
    assert ca["version"] == V and ca["revision"] == 1
    assert ca["close_id"] == "close-cy-a"
    assert ca["detail_digest"] == a["detail_digest"]
    assert ca["report_digest"] == a["report_digest"]
    assert ca["factor_version"] == "grid-2024-v1"
    assert Decimal(ca["total_energy"]) == Decimal("60")
    assert Decimal(ca["total_amount"]) == Decimal("9.00")
    assert Decimal(ca["total_emissions"]) == Decimal("24.000000")
    assert Decimal(ca["carbon_intensity"]) == Decimal("0.400000")
    assert cb["revision"] == 2
    assert Decimal(cb["total_energy"]) == Decimal("120")
    assert Decimal(cb["total_amount"]) == Decimal("18.60")
    assert Decimal(cb["total_emissions"]) == Decimal("60.000000")
    assert Decimal(cb["carbon_intensity"]) == Decimal("0.500000")

    # totals are conserved: exact sums of the stored per-cycle values
    assert Decimal(body["total_energy"]) == Decimal("180")
    assert Decimal(body["total_amount"]) == Decimal("27.60")
    assert Decimal(body["total_emissions"]) == Decimal("84.000000")
    # 84 / 180 = 0.466666... -> 0.466667
    assert Decimal(body["carbon_intensity"]) == Decimal("0.466667")
    assert len(body["statement_digest"]) == 64
    assert body["created_at"]
    assert _count(eng, Statement) == 1


def test_single_cycle_statement(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    body = _create(eng, "st-single", [a]).body
    assert body["cycle_count"] == 1
    assert body["total_energy"] == "60"
    assert body["total_amount"] == "9.00"
    assert body["total_emissions"] == "24.000000"
    assert body["carbon_intensity"] == "0.400000"


def test_twelve_cycles_boundary(tmp_engine):
    """The 1~12 cycle bound: twelve closed cycles aggregate fine."""
    eng, _ = tmp_engine
    # weekend days in July/August 2024 (leisure rule, 10 kWh -> 1.50 each)
    days = ["2024-07-06", "2024-07-07", "2024-07-13", "2024-07-14",
            "2024-07-20", "2024-07-21", "2024-07-27", "2024-07-28",
            "2024-08-03", "2024-08-04", "2024-08-10", "2024-08-11"]
    entries = []
    for i, day in enumerate(days):
        iv = {"start": f"{day}T00:00:00-04:00", "end": f"{day}T01:00:00-04:00",
              "timezone": "America/New_York", "energy": "10"}
        factors = [{"start": f"{day}T04:00:00Z", "end": f"{day}T05:00:00Z",
                    "factor": "0.4"}]
        entries.append(_cycle_entry(
            eng, f"cy-{i:02d}", [("k1", 0, [iv])], f"close-{i}", "1.50",
            factors))
    body = _create(eng, "st-12", entries).body
    assert body["cycle_count"] == 12
    assert len(body["cycles"]) == 12
    assert Decimal(body["total_energy"]) == Decimal("120")
    assert Decimal(body["total_amount"]) == Decimal("18.00")
    assert Decimal(body["total_emissions"]) == Decimal("48.000000")
    assert Decimal(body["carbon_intensity"]) == Decimal("0.400000")


def test_cycles_sorted_by_closed_at(tmp_engine, monkeypatch):
    """Output order follows the closing time, not the request order."""
    eng, _ = tmp_engine
    monkeypatch.setattr(closing, "_now",
                        lambda: "2024-08-02T00:00:00+00:00")
    b = _cycle_b(eng)  # closed "later" (08-02)
    monkeypatch.setattr(closing, "_now",
                        lambda: "2024-08-01T00:00:00+00:00")
    a = _cycle_a(eng)  # closed "earlier" (08-01)
    body = _create(eng, "st-order", [b, a]).body  # request lists b first
    assert [c["cycle_id"] for c in body["cycles"]] == ["cy-a", "cy-b"]
    assert [c["closed_at"] for c in body["cycles"]] == [
        "2024-08-01T00:00:00+00:00", "2024-08-02T00:00:00+00:00"]


def test_same_closed_at_tiebreaks_by_cycle_id(tmp_engine, monkeypatch):
    eng, _ = tmp_engine
    monkeypatch.setattr(closing, "_now",
                        lambda: "2024-08-01T00:00:00+00:00")
    b = _cycle_b(eng)
    a = _cycle_a(eng)
    body = _create(eng, "st-tie", [b, a]).body
    assert [c["cycle_id"] for c in body["cycles"]] == ["cy-a", "cy-b"]


def test_conservation_against_stored_rows(tmp_engine):
    """Statement totals equal the sums of the persisted closure/report
    values (nothing is recomputed from live batches)."""
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    body = _create(eng, "st-conserve", [a, b]).body
    with Session(eng) as s:
        closures = s.scalars(select(CycleClosure)).all()
        reports = s.scalars(select(CarbonReport)).all()
    assert Decimal(body["total_amount"]) == sum(
        (Decimal(c.total_amount) for c in closures), Decimal("0"))
    assert Decimal(body["total_energy"]) == sum(
        (Decimal(c.total_energy) for c in closures), Decimal("0"))
    assert Decimal(body["total_emissions"]) == sum(
        (Decimal(r.total_emissions) for r in reports), Decimal("0"))
    by_cycle = {c["cycle_id"]: c for c in body["cycles"]}
    for c in closures:
        row = by_cycle[c.cycle_id]
        assert row["total_energy"] == c.total_energy
        assert row["total_amount"] == c.total_amount
        assert row["detail_digest"] == c.detail_digest


def test_statement_does_not_touch_cycle_state(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    before = {cid: settlement.get_cycle(cid, engine=eng)
              for cid in ("cy-a", "cy-b")}
    _create(eng, "st-readonly", [a, b])
    after = {cid: settlement.get_cycle(cid, engine=eng)
             for cid in ("cy-a", "cy-b")}
    assert before == after


# ------------------------------------------------------------- idempotency


def test_same_content_replays_statement(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    first = _create(eng, "st-1", [a, b])
    replay = _create(eng, "st-1", [a, b])
    assert replay.replayed is True
    assert replay.body == first.body
    assert _count(eng, Statement) == 1


def test_cycle_order_does_not_affect_replay(tmp_engine):
    """The cycle list order in the request does not change the content
    digest: a reordered request replays the first statement."""
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    first = _create(eng, "st-1", [a, b])
    replay = _create(eng, "st-1", [b, a])
    assert replay.replayed is True
    assert replay.body == first.body
    assert _count(eng, Statement) == 1


def test_same_id_different_cycle_list_conflicts(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    _create(eng, "st-1", [a, b])
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [a])
    assert ei.value.code == "statement_conflict"
    assert ei.value.status_code == 409
    det = ei.value.details
    assert det["existing"]["cycle_count"] == 2
    assert det["received"]["content_digest"] != det["existing"]["content_digest"]
    assert _count(eng, Statement) == 1


def test_same_id_different_digest_conflicts(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    _create(eng, "st-1", [a])
    tampered = dict(a, report_digest="f" * 64)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [tampered])
    assert ei.value.code == "statement_conflict"
    assert _count(eng, Statement) == 1


# ------------------------------------------------------------- rejections


def test_unknown_cycle_404(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [{"cycle_id": "missing",
                               "detail_digest": "a" * 64,
                               "report_digest": "b" * 64}])
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"
    assert _count(eng, Statement) == 0


def test_unclosed_cycle_rejected(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "cy-open", "k1", 0, [A1])
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [{"cycle_id": "cy-open",
                               "detail_digest": "a" * 64,
                               "report_digest": "b" * 64}])
    assert ei.value.code == "cycle_not_closed"
    assert ei.value.status_code == 409
    assert _count(eng, Statement) == 0


def test_closed_cycle_without_carbon_report_rejected(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "cy-nocarbon", "k1", 0, [A1])
    closed = _close(eng, "cy-nocarbon", "close-nc", 1, "9.00").body
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [{"cycle_id": "cy-nocarbon",
                               "detail_digest": closed["detail_digest"],
                               "report_digest": "b" * 64}])
    assert ei.value.code == "carbon_report_missing"
    assert ei.value.status_code == 409
    assert _count(eng, Statement) == 0


def test_detail_digest_mismatch_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    bad = dict(a, detail_digest="f" * 64)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [bad])
    assert ei.value.code == "detail_digest_mismatch"
    assert ei.value.status_code == 409
    assert ei.value.details["expected_detail_digest"] == "f" * 64
    assert ei.value.details["actual_detail_digest"] == a["detail_digest"]
    assert _count(eng, Statement) == 0


def test_report_digest_mismatch_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    bad = dict(a, report_digest="0" * 64)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [bad])
    assert ei.value.code == "report_digest_mismatch"
    assert ei.value.status_code == 409
    assert ei.value.details["actual_report_digest"] == a["report_digest"]
    assert _count(eng, Statement) == 0


def test_duplicate_cycle_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [a, dict(a)])
    assert ei.value.code == "duplicate_cycle"
    assert ei.value.status_code == 400
    assert _count(eng, Statement) == 0


def test_revision_mismatch_between_report_and_closure_rejected(tmp_engine):
    """Carbon report generated at revision 1, then another batch and the
    close at revision 2: the report no longer matches the closure."""
    eng, _ = tmp_engine
    _submit(eng, "cy-e", "k1", 0, [B1])
    _gen(eng, "cy-e", 1, FB)  # report at revision 1 (covers B1's range)
    _submit(eng, "cy-e", "k2", 1, [B2])
    closed = _close(eng, "cy-e", "close-e", 2, "18.60").body
    with Session(eng) as s:
        report = s.get(CarbonReport, "cy-e")
        assert report.revision == 1
        entry = {"cycle_id": "cy-e",
                 "detail_digest": closed["detail_digest"],
                 "report_digest": report.report_digest}
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [entry])
    assert ei.value.code == "revision_mismatch"
    assert ei.value.status_code == 409
    assert ei.value.details["closure_revision"] == 2
    assert ei.value.details["report_revision"] == 1
    assert _count(eng, Statement) == 0


def test_currency_mismatch_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)          # USD
    c = _cycle_c(eng)          # CNY (Shanghai demo version)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [a, c])
    assert ei.value.code == "incompatible_cycles"
    assert ei.value.status_code == 409
    assert ei.value.details["field"] == "currency"
    assert ei.value.details["values"] == {"cy-a": "USD", "cy-c": "CNY"}
    assert _count(eng, Statement) == 0


def test_energy_unit_mismatch_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)          # kWh
    with Session(eng) as s:    # custom version billed in MJ
        v = TariffVersion(code="mj-demo", currency="USD", energy_unit="MJ",
                          description="test")
        rule = RateRule(code="flat", name="flat",
                        effective_from="2024-01-01T00:00:00",
                        effective_to="2025-01-01T00:00:00",
                        timezone="UTC", weekdays="", day_scope="all",
                        time_on="00:00", time_off="24:00")
        rule.tiers.append(Tier(ordinal=1, cumulative_cap=None,
                               unit_price="0.10"))
        v.rules.append(rule)
        s.add(v)
        s.commit()
    iv = {"start": "2024-07-06T10:00:00Z", "end": "2024-07-06T11:00:00Z",
          "timezone": "UTC", "energy": "50"}
    factors = [{"start": "2024-07-06T10:00:00Z",
                "end": "2024-07-06T11:00:00Z", "factor": "0.2"}]
    m = _cycle_entry(eng, "cy-mj", [("k1", 0, [iv])], "close-mj", "5.00",
                     factors, version="mj-demo")
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [a, m])
    assert ei.value.code == "incompatible_cycles"
    assert ei.value.details["field"] == "energy_unit"
    assert ei.value.details["values"] == {"cy-a": "kWh", "cy-mj": "MJ"}
    assert _count(eng, Statement) == 0


def test_cross_cycle_interval_overlap_rejected(tmp_engine):
    """Two cycles whose UTC usage intervals overlap cannot share a
    statement (touching is fine, overlapping is not)."""
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    # same UTC interval as A1, submitted to a different cycle
    d = _cycle_entry(eng, "cy-d", [("k1", 0, [A1])], "close-d", "9.00", FA)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [a, d])
    assert ei.value.code == "cycle_interval_overlap"
    assert ei.value.status_code == 409
    det = ei.value.details
    assert {det["interval_a"]["cycle_id"],
            det["interval_b"]["cycle_id"]} == {"cy-a", "cy-d"}
    assert _count(eng, Statement) == 0


def test_touching_cycles_allowed(tmp_engine):
    """Back-to-back intervals across cycles (end == start) do not count
    as overlapping."""
    eng, _ = tmp_engine
    a = _cycle_a(eng)  # 04:00-05:00 UTC
    nxt = dict(A1, start="2024-07-06T01:00:00-04:00",
               end="2024-07-06T02:00:00-04:00")  # 05:00-06:00 UTC
    factors = [{"start": "2024-07-06T05:00:00Z",
                "end": "2024-07-06T06:00:00Z", "factor": "0.4"}]
    d = _cycle_entry(eng, "cy-d", [("k1", 0, [nxt])], "close-d", "9.00",
                     factors)
    body = _create(eng, "st-touch", [a, d]).body
    assert body["cycle_count"] == 2
    assert Decimal(body["total_energy"]) == Decimal("120")


def test_invalid_digest_format_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [dict(a, detail_digest="not-hex")])
    assert ei.value.code == "invalid_digest"
    assert ei.value.status_code == 400
    with pytest.raises(PricingError) as ei:
        _create(eng, "st-1", [dict(a, report_digest="AB" * 32)])  # uppercase
    assert ei.value.code == "invalid_digest"
    assert _count(eng, Statement) == 0


def test_invalid_statement_id_rejected(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    with pytest.raises(PricingError) as ei:
        _create(eng, "bad id!", [a])
    assert ei.value.code == "invalid_statement_id"
    assert ei.value.status_code == 400
    assert _count(eng, Statement) == 0


# --------------------------------------------------------------- read model


def test_get_statement_not_found(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        statement.get_statement("nope", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "statement_not_found"


def test_get_statement_returns_stored_body(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    body = _create(eng, "st-1", [a, b]).body
    assert statement.get_statement("st-1", engine=eng) == body


# ------------------------------------------------------------- concurrency


def test_concurrent_same_content_creates_one_statement(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    start = threading.Event()
    outcomes: list = []

    def worker():
        start.wait()
        try:
            outcomes.append(_create(eng, "st-race", [a, b]))
        except PricingError as e:
            outcomes.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    oks = [r for r in outcomes if not isinstance(r, PricingError)]
    assert len(oks) == 4, outcomes
    assert sum(1 for r in oks if not r.replayed) == 1
    assert all(r.body == oks[0].body for r in oks)
    assert _count(eng, Statement) == 1


def test_concurrent_different_content_only_one_wins(tmp_engine):
    eng, _ = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    d = _cycle_entry(eng, "cy-d", [("k1", 0, [dict(
        A1, start="2024-07-13T00:00:00-04:00",
        end="2024-07-13T01:00:00-04:00")])], "close-d", "9.00",
        [{"start": "2024-07-13T04:00:00Z", "end": "2024-07-13T05:00:00Z",
          "factor": "0.4"}])
    start = threading.Event()
    outcomes: list = []

    def worker(cycles):
        start.wait()
        try:
            outcomes.append(_create(eng, "st-race", cycles))
        except PricingError as e:
            outcomes.append(e)

    # four distinct (individually valid) contents race for the same id
    variants = [[a], [b], [a, b], [d]]
    threads = [threading.Thread(target=worker, args=(c,)) for c in variants]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    oks = [r for r in outcomes if not isinstance(r, PricingError)]
    errs = [r for r in outcomes if isinstance(r, PricingError)]
    winners = [r for r in oks if not r.replayed]
    assert len(winners) == 1, outcomes
    assert all(e.code == "statement_conflict" for e in errs)
    assert _count(eng, Statement) == 1


# ------------------------------------------------------------- persistence


def test_statement_survives_restart(tmp_engine):
    eng, path = tmp_engine
    a = _cycle_a(eng)
    b = _cycle_b(eng)
    body = _create(eng, "st-1", [a, b]).body
    eng.dispose()

    eng2 = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng2)
    try:
        # statement query after restart
        assert statement.get_statement("st-1", engine=eng2) == body
        # replay after restart returns the first body
        replay = _create(eng2, "st-1", [a, b])
        assert replay.replayed is True
        assert replay.body == body
        # conflicts are still detected after restart
        with pytest.raises(PricingError) as ei:
            _create(eng2, "st-1", [a])
        assert ei.value.code == "statement_conflict"
        with pytest.raises(PricingError) as ei:
            _create(eng2, "st-1", [dict(a, report_digest="0" * 64), b])
        assert ei.value.code == "statement_conflict"
        assert _count(eng2, Statement) == 1
    finally:
        eng2.dispose()


# ------------------------------------------------------------- HTTP surface


def _http_cycle(client, cycle, batches, close_id, total, factors, version=V):
    for key, rev, ivs in batches:
        r = client.post(f"/cycles/{cycle}/batches", json={
            "version": version, "idempotency_key": key,
            "expected_revision": rev,
            "intervals": ivs,
        })
        assert r.status_code == 201, r.text
    r = client.post(f"/cycles/{cycle}/close", json={
        "close_id": close_id, "expected_revision": len(batches),
        "expected_total_amount": total,
    })
    assert r.status_code == 201, r.text
    closed = r.json()
    r = client.post(f"/cycles/{cycle}/carbon", json={
        "factor_version": "grid-2024-v1",
        "expected_revision": len(batches),
        "factors": factors,
    })
    assert r.status_code == 201, r.text
    report = r.json()
    return {"cycle_id": cycle,
            "detail_digest": closed["detail_digest"],
            "report_digest": report["report_digest"]}


def test_http_statement_full_flow():
    client = TestClient(app)
    with client:
        a = _http_cycle(client, "http-st-a", [("k1", 0, [A1])],
                        "close-http-a", "9.00", FA)
        b = _http_cycle(client, "http-st-b",
                        [("k1", 0, [B1]), ("k2", 1, [B2])],
                        "close-http-b", "18.60", FB)

        payload = {"statement_id": "http-st-1", "cycles": [a, b]}
        r = client.post("/statements", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["cycle_count"] == 2
        assert [c["cycle_id"] for c in body["cycles"]] == [
            "http-st-a", "http-st-b"]
        assert body["total_energy"] == "180"
        assert body["total_amount"] == "27.60"
        assert body["total_emissions"] == "84.000000"
        assert body["carbon_intensity"] == "0.466667"
        assert len(body["statement_digest"]) == 64

        # replay -> 200, identical body
        r2 = client.post("/statements", json=payload)
        assert r2.status_code == 200, r2.text
        assert r2.json() == body

        # GET -> same body
        rg = client.get("/statements/http-st-1")
        assert rg.status_code == 200
        assert rg.json() == body

        # same id, different cycle list -> 409 statement_conflict
        r3 = client.post("/statements", json={
            "statement_id": "http-st-1", "cycles": [a]})
        assert r3.status_code == 409
        assert r3.json()["error"]["code"] == "statement_conflict"

        # unknown statement -> 404
        r4 = client.get("/statements/no-such-statement")
        assert r4.status_code == 404
        assert r4.json()["error"]["code"] == "statement_not_found"


def test_http_statement_validation_errors():
    client = TestClient(app)
    with client:
        a = _http_cycle(client, "http-st-err", [("k1", 0, [A1])],
                        "close-http-err", "9.00", FA)

        # digest mismatch -> 409, nothing persisted
        r = client.post("/statements", json={
            "statement_id": "http-st-err-1",
            "cycles": [dict(a, detail_digest="f" * 64)],
        })
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "detail_digest_mismatch"
        assert client.get("/statements/http-st-err-1").status_code == 404

        # duplicate cycle -> 400
        r = client.post("/statements", json={
            "statement_id": "http-st-err-2", "cycles": [a, a]})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "duplicate_cycle"

        # malformed digest -> 400
        r = client.post("/statements", json={
            "statement_id": "http-st-err-3",
            "cycles": [dict(a, report_digest="zz")]})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_digest"

        # illegal statement id -> 400
        r = client.post("/statements", json={
            "statement_id": "bad id", "cycles": [a]})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_statement_id"

        # 0 cycles / 13 cycles -> 422 (schema bounds 1..12)
        r = client.post("/statements", json={
            "statement_id": "http-st-err-4", "cycles": []})
        assert r.status_code == 422
        r = client.post("/statements", json={
            "statement_id": "http-st-err-5",
            "cycles": [a] * 13})
        assert r.status_code == 422

        # unknown cycle -> 404
        r = client.post("/statements", json={
            "statement_id": "http-st-err-6",
            "cycles": [{"cycle_id": "ghost",
                        "detail_digest": "a" * 64,
                        "report_digest": "b" * 64}]})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "cycle_not_found"
