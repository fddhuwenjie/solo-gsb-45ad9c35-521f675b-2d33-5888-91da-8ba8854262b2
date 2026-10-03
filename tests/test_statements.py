"""Tests for consolidated settlement & carbon statements across closed cycles.

Like the closing/carbon tests, these run against a temporary *file-backed*
SQLite database (WAL, real write locks), so concurrent statement creation and
restart/replay semantics exercise the real persistence layer.
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

from app import carbon, closing, settlement, statements  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    CarbonReport,
    RateRule,
    SettlementStatement,
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
    SettlementStatementRequest,
    StatementCycleIn,
)

V = "us-ny-demo-v2024"
# Both days are Saturdays in NY -> only the `leisure` rule
# (tiers: 100 @0.15, 300 @0.18, open @0.22); UTC = local + 4 h.
W1_A = {
    "start": "2024-07-06T00:00:00-04:00",
    "end": "2024-07-06T01:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
W1_B = {
    "start": "2024-07-06T01:00:00-04:00",
    "end": "2024-07-06T02:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
W2 = {
    "start": "2024-07-13T00:00:00-04:00",
    "end": "2024-07-13T02:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
# Wednesday -> dst-workday (100 @0.22 ...), UTC 00:00-01:00 == 20:00-21:00 -04:00.
WED = {
    "start": "2024-07-10T20:00:00-04:00",
    "end": "2024-07-10T21:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "10",
}
FACTORS = {
    "c1": ("2024-07-06T04:00:00Z", "2024-07-06T06:00:00Z", "0.4"),
    "c2": ("2024-07-13T04:00:00Z", "2024-07-13T06:00:00Z", "0.5"),
    "c3": ("2024-07-11T00:00:00Z", "2024-07-11T01:00:00Z", "0.3"),
}


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
    req = type("Req", (), {
        "version": version,
        "idempotency_key": key,
        "expected_revision": rev,
        "intervals": [QuoteIntervalIn(**iv) for iv in intervals],
    })()
    return settlement.submit_batch(cycle, req, engine=eng)


def _close(eng, cycle, rev, total, close_id=None):
    req = CycleCloseRequest(close_id=close_id or f"close-{cycle}",
                            expected_revision=rev,
                            expected_total_amount=Decimal(total))
    return closing.close_cycle(cycle, req, engine=eng)


def _carbon(eng, cycle, rev, factor=None, factor_version="grid-2024-v1"):
    start, end, value = factor or FACTORS[cycle]
    req = CarbonReportRequest(
        factor_version=factor_version, expected_revision=rev,
        factors=[{"start": start, "end": end, "factor": value}])
    return carbon.generate_report(cycle, req, engine=eng)


def _entry(cycle, snap, report):
    return StatementCycleIn(cycle_id=cycle,
                            detail_digest=snap["detail_digest"],
                            report_digest=report["report_digest"])


def _req(entries):
    return SettlementStatementRequest(cycles=list(entries))


def _count(eng, model):
    with Session(eng) as s:
        return len(s.scalars(select(model)).all())


# ------------------------------------------------------------- happy paths


def test_statement_aggregates_two_cycles_sorted_by_closed_at(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])

    res = statements.create_statement(
        "stmt-1", _req([_entry("c2", s2, r2), _entry("c1", s1, r1)]),
        engine=eng)
    assert res.replayed is False
    body = res.body
    assert body["statement_id"] == "stmt-1"
    assert body["currency"] == "USD"
    assert body["energy_unit"] == "kWh"
    assert body["emission_unit"] == "kgCO2e"
    assert body["factor_unit"] == "kgCO2e/kWh"
    assert body["cycle_count"] == 2
    # input order reversed, but output is closed_at ascending
    assert [c["cycle_id"] for c in body["cycles"]] == ["c1", "c2"]
    # totals conserved: 120 + 60 energy; 18.60 + 9.00 amount;
    # 120*0.4 + 60*0.5 = 78 emissions
    assert Decimal(body["total_energy"]) == Decimal("180")
    assert Decimal(body["total_amount"]) == Decimal("27.60")
    assert Decimal(body["total_emissions"]) == Decimal("78.000000")
    # 78 / 180 = 0.433333... -> 0.433333 at 6 dp
    assert body["carbon_intensity"] == "0.433333"
    c1, c2 = body["cycles"]
    assert c1["total_energy"] == "120"
    assert c1["total_amount"] == "18.60"
    assert c1["total_emissions"] == "48.000000"
    assert c1["carbon_intensity"] == "0.400000"
    assert c1["revision"] == 2
    assert c1["version"] == V
    assert c1["factor_version"] == "grid-2024-v1"
    assert c2["total_emissions"] == "30.000000"
    assert c2["carbon_intensity"] == "0.500000"
    # per-cycle sums reconcile exactly to totals
    assert sum(Decimal(c["total_amount"]) for c in body["cycles"]) == Decimal(
        body["total_amount"])
    assert sum(Decimal(c["total_emissions"]) for c in body["cycles"]) == Decimal(
        body["total_emissions"])
    assert sum(Decimal(c["total_energy"]) for c in body["cycles"]) == Decimal(
        body["total_energy"])
    assert len(body["statement_digest"]) == 64
    assert body["created_at"]
    assert _count(eng, SettlementStatement) == 1


def test_statement_values_come_from_snapshots_not_live_recompute(tmp_engine):
    """Amount/energy come from the closure snapshot and emissions from the
    carbon report; the live cycle rows are not re-priced."""
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    res = statements.create_statement(
        "stmt-1", _req([_entry("c1", s1, r1)]), engine=eng).body
    (cycle,) = res["cycles"]
    assert cycle["total_amount"] == s1["total_amount"]
    assert cycle["total_energy"] == s1["total_energy"]
    assert cycle["total_emissions"] == r1["total_emissions"]
    assert cycle["detail_digest"] == s1["detail_digest"]
    assert cycle["report_digest"] == r1["report_digest"]
    assert Decimal(res["total_amount"]) == Decimal("18.60")
    assert Decimal(res["total_emissions"]) == Decimal("48.000000")


def test_statement_digest_is_deterministic(tmp_engine):
    """The digest covers deterministic business content (excluding
    statement_id / created_at): it is stable on replay, equals an independent
    canonical-JSON recomputation, and identical content under a different
    statement id yields the same digest."""
    import hashlib

    from app.importer import canonical_json

    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    entries = [_entry("c2", s2, r2), _entry("c1", s1, r1)]
    first = statements.create_statement(
        "a", _req(entries), engine=eng).body
    replay = statements.create_statement(
        "a", _req(reversed(entries)), engine=eng).body
    assert replay == first  # byte-identical replay incl. digest/created_at

    deterministic = {k: v for k, v in first.items()
                     if k not in ("created_at", "statement_id",
                                  "statement_digest")}
    expected = hashlib.sha256(
        canonical_json(deterministic).encode("utf-8")).hexdigest()
    assert first["statement_digest"] == expected

    other = statements.create_statement(
        "b", _req(entries), engine=eng).body
    assert other["statement_digest"] == first["statement_digest"]
    assert other["cycles"] == first["cycles"]


def test_single_cycle_statement(tmp_engine):
    eng, _ = tmp_engine
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    body = statements.create_statement(
        "solo", _req([_entry("c2", s2, r2)]), engine=eng).body
    assert body["cycle_count"] == 1
    assert Decimal(body["total_amount"]) == Decimal("9.00")
    assert Decimal(body["total_emissions"]) == Decimal("30.000000")
    assert body["carbon_intensity"] == "0.500000"


# ------------------------------------------------------------- idempotency


def test_same_content_replays_statement(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    req1 = _req([_entry("c1", s1, r1), _entry("c2", s2, r2)])
    first = statements.create_statement("stmt-1", req1, engine=eng)
    assert first.replayed is False
    # same id, same content, different input order -> still a replay
    req2 = _req([_entry("c2", s2, r2), _entry("c1", s1, r1)])
    replay = statements.create_statement("stmt-1", req2, engine=eng)
    assert replay.replayed is True
    assert replay.body == first.body
    assert _count(eng, SettlementStatement) == 1


def test_same_id_different_cycle_list_conflicts(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    statements.create_statement(
        "stmt-1", _req([_entry("c1", s1, r1)]), engine=eng)
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "stmt-1",
            _req([_entry("c1", s1, r1), _entry("c2", s2, r2)]),
            engine=eng)
    assert ei.value.code == "statement_conflict"
    assert ei.value.status_code == 409
    det = ei.value.details
    assert det["existing"]["request_digest"] != det["received"]["request_digest"]
    assert [c["cycle_id"] for c in det["received"]["cycles"]] == ["c1", "c2"]
    assert _count(eng, SettlementStatement) == 1


def test_same_id_digest_conflict(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    statements.create_statement(
        "stmt-1", _req([_entry("c1", s1, r1)]), engine=eng)
    wrong = StatementCycleIn(
        cycle_id="c1",
        detail_digest="a" * 64,
        report_digest=r1["report_digest"])
    # the tampered request is rejected by validation before even reaching the
    # idempotency branch; now use a valid-but-different statement with the
    # same id after the stored one exists
    with pytest.raises(PricingError) as ei:
        statements.create_statement("stmt-1", _req([wrong]), engine=eng)
    assert ei.value.code == "statement_conflict"
    assert _count(eng, SettlementStatement) == 1


# ------------------------------------------------------------- rejections


def test_duplicate_cycle_rejected_and_writes_nothing(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    entry = _entry("c1", s1, r1)
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "stmt-1", _req([entry, StatementCycleIn(**entry.model_dump())]),
            engine=eng)
    assert ei.value.code == "duplicate_cycle"
    assert ei.value.status_code == 400
    assert _count(eng, SettlementStatement) == 0


def test_unknown_cycle_404(tmp_engine):
    eng, _ = tmp_engine
    entry = StatementCycleIn(cycle_id="missing",
                             detail_digest="a" * 64,
                             report_digest="b" * 64)
    with pytest.raises(PricingError) as ei:
        statements.create_statement("stmt-1", _req([entry]), engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"
    assert _count(eng, SettlementStatement) == 0


def test_open_cycle_rejected(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "open-1", "k1", 0, [W1_A])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    # no carbon report/closure for open-1: craft digests; status check fires
    entry = StatementCycleIn(cycle_id="open-1",
                             detail_digest="a" * 64,
                             report_digest="b" * 64)
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "stmt-1", _req([entry, _entry("c2", s2, r2)]), engine=eng)
    assert ei.value.code == "cycle_not_closed"
    assert ei.value.status_code == 409
    assert _count(eng, SettlementStatement) == 0


def test_missing_carbon_report_rejected(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "nocarbon", "k1", 0, [W1_A])
    snap = _close(eng, "nocarbon", 1, "9.00").body
    entry = StatementCycleIn(cycle_id="nocarbon",
                             detail_digest=snap["detail_digest"],
                             report_digest="b" * 64)
    with pytest.raises(PricingError) as ei:
        statements.create_statement("stmt-1", _req([entry]), engine=eng)
    assert ei.value.code == "carbon_report_missing"
    assert ei.value.status_code == 409
    assert _count(eng, SettlementStatement) == 0


def test_closure_report_revision_mismatch_rejected(tmp_engine):
    """A carbon report frozen at a different revision than the closure is
    rejected (this state is crafted directly at the storage layer)."""
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    with Session(eng) as s:
        report = s.get(CarbonReport, "c1")
        report.revision = 99
        s.commit()
    entry = StatementCycleIn(cycle_id="c1",
                             detail_digest=s1["detail_digest"],
                             report_digest=r1["report_digest"])
    with pytest.raises(PricingError) as ei:
        statements.create_statement("stmt-1", _req([entry]), engine=eng)
    assert ei.value.code == "closure_report_revision_mismatch"
    assert ei.value.details == {
        "cycle_id": "c1", "closure_revision": 2, "report_revision": 99}
    assert _count(eng, SettlementStatement) == 0


def test_detail_digest_mismatch_rejected(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    entry = StatementCycleIn(cycle_id="c1",
                             detail_digest="a" * 64,
                             report_digest=r1["report_digest"])
    with pytest.raises(PricingError) as ei:
        statements.create_statement("stmt-1", _req([entry]), engine=eng)
    assert ei.value.code == "detail_digest_mismatch"
    assert ei.value.status_code == 409
    assert ei.value.details["expected"] == "a" * 64
    assert ei.value.details["actual"] == s1["detail_digest"]
    assert _count(eng, SettlementStatement) == 0


def test_report_digest_mismatch_rejected(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    entry = StatementCycleIn(cycle_id="c1",
                             detail_digest=s1["detail_digest"],
                             report_digest="b" * 64)
    with pytest.raises(PricingError) as ei:
        statements.create_statement("stmt-1", _req([entry]), engine=eng)
    assert ei.value.code == "report_digest_mismatch"
    assert ei.value.details["actual"] == r1["report_digest"]
    assert _count(eng, SettlementStatement) == 0


def test_cross_cycle_overlap_rejected(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    # cX covers the exact same UTC window as c1 (04:00-06:00 on 2024-07-06).
    sx, rx = _closed_carbon_cycle(
        eng, "cX", [dict(W1_A), dict(W1_B)], factor_key="c1")
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "stmt-1", _req([_entry("c1", s1, r1),
                            _entry("cX", sx, rx)]), engine=eng)
    assert ei.value.code == "cycle_interval_overlap"
    assert ei.value.status_code == 409
    det = ei.value.details
    assert {det["interval_a"]["cycle_id"],
            det["interval_b"]["cycle_id"]} == {"c1", "cX"}
    assert _count(eng, SettlementStatement) == 0


def test_touching_cycles_allowed(tmp_engine):
    """Cycles whose UTC intervals merely touch (end == next start) are fine."""
    eng, _ = tmp_engine
    left_iv = {"start": "2024-07-06T00:00:00-04:00",
               "end": "2024-07-06T01:00:00-04:00",
               "timezone": "America/New_York", "energy": "60"}
    right_iv = {"start": "2024-07-06T01:00:00-04:00",
                "end": "2024-07-06T02:00:00-04:00",
                "timezone": "America/New_York", "energy": "60"}
    FACTORS.setdefault("left",
                       ("2024-07-06T04:00:00Z",
                        "2024-07-06T05:00:00Z", "0.4"))
    FACTORS.setdefault("right",
                       ("2024-07-06T05:00:00Z",
                        "2024-07-06T06:00:00Z", "0.6"))
    s_l, r_l = _closed_carbon_cycle(
        eng, "left", [left_iv], factor_key="left")
    s_r, r_r = _closed_carbon_cycle(
        eng, "right", [right_iv], factor_key="right")
    body = statements.create_statement(
        "touch", _req([_entry("left", s_l, r_l),
                       _entry("right", s_r, r_r)]), engine=eng).body
    assert Decimal(body["total_energy"]) == Decimal("120")
    # 60*0.4 + 60*0.6 = 60
    assert Decimal(body["total_emissions"]) == Decimal("60.000000")


def test_incompatible_currency_rejected(tmp_engine):
    eng, _ = tmp_engine
    _seed_other_version(eng, code="usd-therm", currency="USD",
                        energy_unit="therm")
    s_usd, r_usd = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s_wed, r_wed = _closed_carbon_cycle(
        eng, "c3", [WED], version="usd-therm", factor_key="c3")
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "stmt-1",
            _req([_entry("c1", s_usd, r_usd),
                  _entry("c3", s_wed, r_wed)]), engine=eng)
    assert ei.value.code == "incompatible_cycles"
    assert ei.value.status_code == 409
    assert ei.value.details["field"] == "能量单位"
    assert ei.value.details["actual"] == "therm"
    assert _count(eng, SettlementStatement) == 0


def test_incompatible_energy_unit_via_cny(tmp_engine):
    """The seeded cn-shanghai version shares kWh but uses CNY: currency
    incompatibility is rejected as well."""
    eng, _ = tmp_engine
    iv = {"start": "2024-07-06T00:00:00-04:00",
          "end": "2024-07-06T01:00:00-04:00",
          "timezone": "America/New_York", "energy": "10"}
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A])
    s_cn, r_cn = _closed_carbon_cycle(
        eng, "cn1", [iv], version="cn-shanghai-demo-v2024",
        factor_key="c1")
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "stmt-1",
            _req([_entry("c1", s1, r1),
                  _entry("cn1", s_cn, r_cn)]), engine=eng)
    assert ei.value.code == "incompatible_cycles"
    assert ei.value.details["field"] == "币种"
    assert ei.value.details["actual"] == "CNY"
    assert _count(eng, SettlementStatement) == 0


def _seed_other_version(eng, *, code, currency, energy_unit):
    """Seed a minimal valid one-rule version directly in the temp engine."""
    with Session(eng) as s:
        v = TariffVersion(code=code, currency=currency,
                          energy_unit=energy_unit, description="test")
        rule = RateRule(
            code="flat", name="flat",
            effective_from="2024-01-01T00:00:00",
            effective_to="2025-01-01T00:00:00",
            timezone="America/New_York", weekdays="", day_scope="all",
            time_on="00:00", time_off="24:00")
        rule.tiers.append(Tier(ordinal=1, cumulative_cap=None,
                               unit_price="0.10"))
        v.rules.append(rule)
        s.add(v)
        s.commit()


def _closed_carbon_cycle(eng, cycle, intervals, *, version=V,
                         factor_key=None):
    cur = 0
    for k, iv in enumerate(intervals, start=1):
        _submit(eng, cycle, f"k{k}", cur, [iv], version=version)
        cur += 1
    total = settlement.get_cycle(cycle, engine=eng)["total_amount"]
    snap = _close(eng, cycle, cur, total).body
    key = factor_key or cycle
    if key not in FACTORS:
        raise KeyError(f"no factors registered for {cycle!r}")
    report = _carbon(eng, cycle, cur, factor=FACTORS[key]).body
    return snap, report


def test_rejection_writes_no_partial_state(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    # unknown cycle -> nothing stored even though c1 is valid
    bad = StatementCycleIn(cycle_id="ghost",
                           detail_digest="a" * 64,
                           report_digest="b" * 64)
    with pytest.raises(PricingError):
        statements.create_statement(
            "stmt-1", _req([_entry("c1", s1, r1), bad]), engine=eng)
    assert _count(eng, SettlementStatement) == 0


# --------------------------------------------------------------- read model


def test_get_statement_not_found(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        statements.get_statement("nope", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "statement_not_found"


def test_get_statement_returns_stored_body(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    body = statements.create_statement(
        "stmt-1", _req([_entry("c1", s1, r1)]), engine=eng).body
    assert statements.get_statement("stmt-1", engine=eng) == body


def test_invalid_statement_id(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    with pytest.raises(PricingError) as ei:
        statements.create_statement(
            "../x", _req([_entry("c1", s1, r1)]), engine=eng)
    assert ei.value.code == "invalid_statement_id"
    assert ei.value.status_code == 400


# ------------------------------------------------------------- concurrency


def test_concurrent_same_content_creates_one_statement(tmp_engine):
    eng, _ = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    req = _req([_entry("c1", s1, r1), _entry("c2", s2, r2)])
    start = threading.Event()
    outcomes: list = []

    def worker():
        start.wait()
        try:
            outcomes.append(
                statements.create_statement("race", req, engine=eng))
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
    assert _count(eng, SettlementStatement) == 1


# ------------------------------------------------------------- persistence


def test_statement_survives_restart(tmp_engine):
    eng, path = tmp_engine
    s1, r1 = _closed_carbon_cycle(eng, "c1", [W1_A, W1_B])
    s2, r2 = _closed_carbon_cycle(eng, "c2", [W2])
    req = _req([_entry("c1", s1, r1), _entry("c2", s2, r2)])
    body = statements.create_statement("stmt-1", req, engine=eng).body
    eng.dispose()

    eng2 = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng2)
    try:
        # GET after restart
        assert statements.get_statement("stmt-1", engine=eng2) == body
        # same-content replay after restart returns the stored body
        replay = statements.create_statement("stmt-1", req, engine=eng2)
        assert replay.replayed is True
        assert replay.body == body
        # changed cycle list -> conflict even after restart
        with pytest.raises(PricingError) as ei:
            statements.create_statement(
                "stmt-1", _req([_entry("c1", s1, r1)]), engine=eng2)
        assert ei.value.code == "statement_conflict"
    finally:
        eng2.dispose()


# ------------------------------------------------------------- HTTP surface


def test_http_statement_full_flow():
    client = TestClient(app)
    with client:
        def batch(cid, key, rev, iv):
            r = client.post(f"/cycles/{cid}/batches", json={
                "version": V, "idempotency_key": key,
                "expected_revision": rev, "intervals": [iv]})
            assert r.status_code == 201, r.text
            return r.json()

        b1 = batch("http-stmt-c1", "k1", 0, W1_A)
        batch("http-stmt-c1", "k2", 1, W1_B)
        b2 = batch("http-stmt-c2", "k1", 0, W2)
        for cid, rev, total in (
            ("http-stmt-c1", 2, "18.60"),
            ("http-stmt-c2", 1, b2["total_amount"]),
        ):
            r = client.post(f"/cycles/{cid}/close", json={
                "close_id": f"close-{cid}", "expected_revision": rev,
                "expected_total_amount": total})
            assert r.status_code == 201, r.text

        def factors(cid, start, end):
            r = client.post(f"/cycles/{cid}/carbon", json={
                "factor_version": "grid-2024-v1",
                "expected_revision": (2 if cid.endswith("c1") else 1),
                "factors": [{"start": start, "end": end, "factor": "0.4"}
                            if cid.endswith("c1")
                            else {"start": start, "end": end, "factor": "0.5"}]})
            assert r.status_code == 201, r.text
            return r.json()

        rep1 = factors("http-stmt-c1",
                       "2024-07-06T04:00:00Z", "2024-07-06T06:00:00Z")
        rep2 = factors("http-stmt-c2",
                       "2024-07-13T04:00:00Z", "2024-07-13T06:00:00Z")
        snap1 = client.get("/cycles/http-stmt-c1/close").json()
        snap2 = client.get("/cycles/http-stmt-c2/close").json()

        payload = {"cycles": [
            {"cycle_id": "http-stmt-c2",
             "detail_digest": snap2["detail_digest"],
             "report_digest": rep2["report_digest"]},
            {"cycle_id": "http-stmt-c1",
             "detail_digest": snap1["detail_digest"],
             "report_digest": rep1["report_digest"]},
        ]}
        r = client.post("/statements/stmt-http-1", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        assert [c["cycle_id"] for c in body["cycles"]] == [
            "http-stmt-c1", "http-stmt-c2"]
        assert body["total_amount"] == "27.60"
        assert body["total_emissions"] == "78.000000"
        assert body["carbon_intensity"] == "0.433333"

        # replay -> 200 identical
        r2 = client.post("/statements/stmt-http-1", json=payload)
        assert r2.status_code == 200
        assert r2.json() == body

        # GET -> same body
        rg = client.get("/statements/stmt-http-1")
        assert rg.status_code == 200
        assert rg.json() == body


def test_http_statement_errors():
    client = TestClient(app)
    with client:
        r = client.get("/statements/no-such")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "statement_not_found"

        # empty cycle list -> 422
        r = client.post("/statements/s1", json={"cycles": []})
        assert r.status_code == 422

        # malformed digest -> 422
        cid = "http-stmt-err-c1"
        rb = client.post(f"/cycles/{cid}/batches", json={
            "version": V, "idempotency_key": "k1", "expected_revision": 0,
            "intervals": [W1_A]})
        assert rb.status_code == 201
        r = client.post("/statements/s2", json={"cycles": [
            {"cycle_id": cid, "detail_digest": "xyz",
             "report_digest": "b" * 64}]})
        assert r.status_code == 422

        # open cycle -> 409 cycle_not_closed
        r = client.post("/statements/s3", json={"cycles": [
            {"cycle_id": cid, "detail_digest": "a" * 64,
             "report_digest": "b" * 64}]})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "cycle_not_closed"

        # duplicate cycle -> 400 duplicate_cycle
        dup = {"cycle_id": cid, "detail_digest": "a" * 64,
               "report_digest": "b" * 64}
        r = client.post("/statements/s4", json={"cycles": [dup, dict(dup)]})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "duplicate_cycle"

        # invalid statement id -> 400
        r = client.post("/statements/bad id", json={"cycles": [dup]})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_statement_id"
