"""跨多个已关账账期的综合结算与碳排声明（不可变）。

调用方提交唯一 ``statement_id`` 与 1～12 个账期条目，每条携带
``cycle_id`` 及期望的 ``detail_digest``（关账明细摘要）与
``report_digest``（碳排报告摘要）。整个汇总在**同一个 SQLite 事务**内
完成读取与校验：

1. 先按 ``statement_id`` 判幂等：同 id 同账期列表/摘要重放首次声明
  （HTTP 200，不写库）；换 cycle 列表或摘要 → 409 ``statement_conflict``。
2. 每个账期必须存在、处于 ``closed``，且 :class:`CycleClosure` 与
   :class:`CarbonReport` 均存在；任一缺失即整体拒绝。
3. 碳排报告的 revision 必须与关账 revision 一致；客户端期望的
   ``detail_digest`` / ``report_digest`` 必须与存储的逐字节相等。
4. 各账期的规则币种与能量单位必须一致；各账期在 **UTC 时间线**上的
   用量区间互不重叠（首尾相接允许），跨账期重叠同样整体拒绝。

校验通过后，所有数字都**直接取自不可变关账快照与碳排报告**（金额取自
关账快照、用量取自关账快照、排放取自碳排报告），绝不从实时批次重新
计价：按关账时间（``closed_at``）升序输出逐账期总用量、总金额、总排放
与碳强度（总排放 ÷ 总用量），并给出三者的合计与综合碳强度；金额、用量、
排放分别严格守恒。最后对规范 JSON 取 SHA-256（``statement_digest``），
在同一事务内写入不可变声明。任何失败整笔回滚，不留下任何声明数据。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from .carbon import EMISSION_UNIT, FACTOR_UNIT
from .database import (
    BillingCycle,
    CarbonReport,
    CycleClosure,
    SettlementStatement,
    engine as default_engine,
)
from .importer import canonical_json
from .pricing import Q2, Q6, PricingError
from .schemas import SettlementStatementRequest

STATEMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

# 与关账/碳排相同的写锁竞争重试策略：延迟事务升级写锁撞上 SQLITE_BUSY
# 时，在全新事务里重读声明行后整笔重跑。
_MAX_LOCK_RETRIES = 20


@dataclass(frozen=True)
class StatementResult:
    body: dict
    replayed: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_statement_id(statement_id: str) -> None:
    if not STATEMENT_ID_RE.fullmatch(statement_id):
        raise PricingError(
            f"非法 statement_id: {statement_id!r}；只允许字母/数字开头，"
            "后跟至多 127 个字母、数字、下划线或连字符",
            code="invalid_statement_id", status_code=400,
            details={"statement_id": statement_id})


# --------------------------------------------------------------------- errors


def _cycle_not_found(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期不存在: {cycle_id}",
        code="cycle_not_found", status_code=404,
        details={"cycle_id": cycle_id})


def _cycle_not_closed(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 尚未关账：综合声明只能汇总已关账且已有不可变"
        "碳排报告的账期",
        code="cycle_not_closed", status_code=409,
        details={"cycle_id": cycle_id})


def _carbon_report_missing(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 缺少不可变碳排报告：综合声明要求每个账期都已"
        "生成碳排快照",
        code="carbon_report_missing", status_code=409,
        details={"cycle_id": cycle_id})


def _revision_mismatch(cycle_id: str, closure_revision: int,
                       report_revision: int) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的碳排报告 revision（{report_revision}）与关账 "
        f"revision（{closure_revision}）不一致，拒绝汇总",
        code="closure_report_revision_mismatch", status_code=409,
        details={"cycle_id": cycle_id,
                 "closure_revision": closure_revision,
                 "report_revision": report_revision})


def _detail_digest_mismatch(cycle_id: str, expected: str,
                            actual: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 detail_digest 与关账快照不一致：期望 "
        f"{expected}，存储为 {actual}，拒绝汇总",
        code="detail_digest_mismatch", status_code=409,
        details={"cycle_id": cycle_id, "expected": expected, "actual": actual})


def _report_digest_mismatch(cycle_id: str, expected: str,
                            actual: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 report_digest 与碳排报告不一致：期望 "
        f"{expected}，存储为 {actual}，拒绝汇总",
        code="report_digest_mismatch", status_code=409,
        details={"cycle_id": cycle_id, "expected": expected, "actual": actual})


def _incompatible(field: str, expected: str, cycle_id: str,
                  actual: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的{field}为 {actual!r}，与首个账期的 {expected!r} "
        "不一致：综合声明只能汇总币种与能量单位完全一致的账期",
        code="incompatible_cycles", status_code=409,
        details={"field": field, "expected": expected,
                 "cycle_id": cycle_id, "actual": actual})


def _overlap_error(a: dict, b: dict) -> PricingError:
    return PricingError(
        f"账期 {a['cycle_id']} 与 {b['cycle_id']} 的用量区间在 UTC 时间线"
        f"上重叠（{a['start']} ~ {a['end']} 与 {b['start']} ~ {b['end']}）；"
        "综合声明的各账期区间必须互不重叠（首尾相接允许）",
        code="cycle_interval_overlap", status_code=409,
        details={"interval_a": a, "interval_b": b})


def _statement_conflict(statement_id: str, existing: SettlementStatement,
                        request_digest: str,
                        cycles: list[dict]) -> PricingError:
    return PricingError(
        f"statement_id {statement_id!r} 已用于内容不同的综合声明：同 "
        "statement_id 仅允许相同账期列表与相同期望摘要的重放，换 cycle "
        "列表或摘要冲突",
        code="statement_conflict", status_code=409,
        details={"statement_id": statement_id,
                 "existing": {
                     "request_digest": existing.request_digest,
                     "statement_digest": existing.statement_digest,
                     "cycle_count": existing.cycle_count,
                 },
                 "received": {
                     "request_digest": request_digest,
                     "cycles": cycles,
                 }})


# ------------------------------------------------------------- normalization


def _normalize_cycles(req: SettlementStatementRequest) -> list[dict]:
    """规范账期列表：按 cycle_id 升序排序（幂等摘要与输入顺序无关）。

    重复 cycle_id 在进入事务前直接拒绝（400 ``duplicate_cycle``）。
    """
    seen: set[str] = set()
    for item in req.cycles:
        if item.cycle_id in seen:
            raise PricingError(
                f"综合声明的账期列表中 cycle_id 重复: {item.cycle_id!r}",
                code="duplicate_cycle", status_code=400,
                details={"cycle_id": item.cycle_id})
        seen.add(item.cycle_id)
    return [
        {"cycle_id": item.cycle_id,
         "detail_digest": item.detail_digest,
         "report_digest": item.report_digest}
        for item in sorted(req.cycles, key=lambda c: c.cycle_id)
    ]


def _request_digest(cycles: list[dict]) -> str:
    """幂等内容摘要：规范后的账期列表（cycle_id + 两个期望摘要）。

    ``statement_id`` 是主键而非内容，刻意不参与摘要；列表输入顺序与重复
    已在规范化阶段处理。
    """
    doc = {"cycles": cycles}
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- main flow


def create_statement(statement_id: str, req: SettlementStatementRequest, *,
                     engine=None) -> StatementResult:
    """汇总已关账账期生成不可变综合声明，返回 ``(声明体, 是否幂等重放)``。

    任何校验失败都抛出 :class:`PricingError` 且整笔事务回滚——不写入
    声明。``engine`` 可注入（测试用临时文件库），默认应用全局引擎。
    """
    validate_statement_id(statement_id)
    cycles = _normalize_cycles(req)
    request_digest = _request_digest(cycles)
    eng = engine or default_engine

    attempt = 0
    while True:
        attempt += 1
        try:
            return _create_once(eng, statement_id, cycles, request_digest)
        except OperationalError as exc:  # SQLITE_BUSY write-lock race
            if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                raise
            # 全新事务重读声明行后整笔重跑。


def _is_lock_error(exc: OperationalError) -> bool:
    msg = str(exc.orig).lower() if exc.orig is not None else ""
    return ("database is locked" in msg
            or "database table is locked" in msg)


def _create_once(eng, statement_id: str, cycles: list[dict],
                 request_digest: str) -> StatementResult:
    with Session(eng) as session:
        try:
            with session.begin():
                # 1) 幂等优先：同 statement_id 重放/冲突先于一切汇总校验。
                existing = session.get(SettlementStatement, statement_id)
                if existing is not None:
                    return _replay_or_conflict(
                        statement_id, existing, request_digest, cycles)

                # 2) 同一事务内依次读取每个账期的关账快照与碳排报告，
                #    状态/缺失/revision/摘要任一不符即整体拒绝。
                loaded = [
                    _load_cycle(session, item) for item in cycles
                ]

                # 3) 规则币种与能量单位必须跨账期一致。
                currency = loaded[0]["currency"]
                energy_unit = loaded[0]["energy_unit"]
                for item in loaded[1:]:
                    if item["currency"] != currency:
                        raise _incompatible(
                            "币种", currency, item["cycle_id"],
                            item["currency"])
                    if item["energy_unit"] != energy_unit:
                        raise _incompatible(
                            "能量单位", energy_unit, item["cycle_id"],
                            item["energy_unit"])

                # 4) 各账期 UTC 用量区间互不重叠（跨账期同样禁止）。
                _ensure_no_cross_cycle_overlap(loaded)

                # 5) 全部数字取自不可变快照/报告，按关账时间稳定排序输出；
                #    金额、用量、排放分别守恒，绝不从实时批次重算。
                created_at = _now()
                body = _statement_body(
                    statement_id=statement_id, loaded=loaded,
                    currency=currency, energy_unit=energy_unit,
                    created_at=created_at)

                session.add(SettlementStatement(
                    statement_id=statement_id,
                    request_digest=request_digest,
                    statement_digest=body["statement_digest"],
                    cycle_count=len(loaded),
                    currency=currency,
                    energy_unit=energy_unit,
                    emission_unit=EMISSION_UNIT,
                    total_energy=body["total_energy"],
                    total_amount=body["total_amount"],
                    total_emissions=body["total_emissions"],
                    carbon_intensity=body["carbon_intensity"],
                    statement_json=json.dumps(body, ensure_ascii=False),
                    created_at=created_at,
                ))
                return StatementResult(body, False)

        except IntegrityError as exc:
            # 主键冲突（并发同 statement_id）：只读重查归类为重放 / 冲突，
            # 绝不写任何数据。
            return _classify_integrity_error(
                eng, statement_id, cycles, request_digest, exc)


def _load_cycle(session: Session, item: dict) -> dict:
    """事务内读取并校验单个账期的关账快照与碳排报告。"""
    cycle_id = item["cycle_id"]
    cycle = session.get(BillingCycle, cycle_id)
    if cycle is None:
        raise _cycle_not_found(cycle_id)
    if cycle.status != "closed":
        raise _cycle_not_closed(cycle_id)

    closure = session.scalar(
        select(CycleClosure).where(CycleClosure.cycle_id == cycle_id))
    if closure is None:
        # 账期状态为 closed 却无快照属于理论不可达的库不一致。
        raise PricingError(
            f"账期 {cycle_id} 已关账但缺少关账快照",
            code="closure_not_found", status_code=404,
            details={"cycle_id": cycle_id})
    report = session.get(CarbonReport, cycle_id)
    if report is None:
        raise _carbon_report_missing(cycle_id)

    if closure.revision != report.revision:
        raise _revision_mismatch(
            cycle_id, closure.revision, report.revision)
    if closure.detail_digest != item["detail_digest"]:
        raise _detail_digest_mismatch(
            cycle_id, item["detail_digest"], closure.detail_digest)
    if report.report_digest != item["report_digest"]:
        raise _report_digest_mismatch(
            cycle_id, item["report_digest"], report.report_digest)

    snapshot = json.loads(closure.snapshot_json)
    report_body = json.loads(report.report_json)
    return {
        "cycle_id": cycle_id,
        "revision": closure.revision,
        "version": snapshot["version"],
        "factor_version": report.factor_version,
        "currency": snapshot["currency"],
        "energy_unit": snapshot["energy_unit"],
        "detail_digest": closure.detail_digest,
        "report_digest": report.report_digest,
        "closed_at": snapshot["closed_at"],
        "total_energy": closure.total_energy,
        "total_amount": closure.total_amount,
        "total_emissions": report.total_emissions,
        "intervals": snapshot["intervals"],
        "emission_unit": report_body.get("emission_unit", EMISSION_UNIT),
    }


def _ensure_no_cross_cycle_overlap(loaded: list[dict]) -> None:
    """收集各账期关账快照中的 UTC 区间并检测跨账期重叠。"""
    spans: list[dict] = []
    for item in loaded:
        for iv in item["intervals"]:
            spans.append({
                "cycle_id": item["cycle_id"],
                "seq": iv["seq"],
                "revision": iv["revision"],
                "start": iv["start"],
                "end": iv["end"],
            })
    spans.sort(key=lambda s: (_parse_ts(s["start"]), _parse_ts(s["end"]),
                              s["cycle_id"], s["seq"]))
    for k in range(1, len(spans)):
        a, b = spans[k - 1], spans[k]
        if _parse_ts(b["start"]) < _parse_ts(a["end"]):
            raise _overlap_error(a, b)


# -------------------------------------------------------------- aggregating


def _intensity(emissions: Decimal, energy: Decimal) -> Decimal:
    return (emissions / energy).quantize(Q6, rounding=ROUND_HALF_UP)


def _statement_body(*, statement_id: str, loaded: list[dict],
                    currency: str, energy_unit: str,
                    created_at: str) -> dict:
    ordered = sorted(
        loaded,
        key=lambda c: (_parse_ts(c["closed_at"]), c["cycle_id"]))

    cycle_docs: list[dict] = []
    total_energy = Decimal("0")
    total_amount = Decimal("0")
    total_emissions = Decimal("0")
    for c in ordered:
        energy = Decimal(c["total_energy"])
        amount = Decimal(c["total_amount"]).quantize(Q2)
        emissions = Decimal(c["total_emissions"]).quantize(Q6)
        total_energy += energy
        total_amount += amount
        total_emissions += emissions
        cycle_docs.append({
            "cycle_id": c["cycle_id"],
            "revision": c["revision"],
            "version": c["version"],
            "factor_version": c["factor_version"],
            "detail_digest": c["detail_digest"],
            "report_digest": c["report_digest"],
            "closed_at": c["closed_at"],
            "total_energy": format(energy.normalize(), "f"),
            "total_amount": str(amount),
            "total_emissions": str(emissions),
            "carbon_intensity": str(_intensity(emissions, energy)),
        })

    total_amount = total_amount.quantize(Q2)
    total_emissions = total_emissions.quantize(Q6)
    # 与关账 detail_digest / 碳排 report_digest 同约定：声明摘要覆盖确定性
    # 业务内容，statement_id 是幂等主键、created_at 是生成时间，均不参与。
    content = {
        "currency": currency,
        "energy_unit": energy_unit,
        "emission_unit": EMISSION_UNIT,
        "factor_unit": FACTOR_UNIT,
        "total_energy": format(total_energy.normalize(), "f"),
        "total_amount": str(total_amount),
        "total_emissions": str(total_emissions),
        "carbon_intensity": str(_intensity(total_emissions, total_energy)),
        "cycle_count": len(cycle_docs),
        "cycles": cycle_docs,
    }
    # 声明摘要只覆盖确定性内容（statement_id 与 created_at 不参与），供离线
    # 核对；同内容换 statement_id 得到相同摘要。
    digest = hashlib.sha256(
        canonical_json(content).encode("utf-8")).hexdigest()
    return {
        "statement_id": statement_id,
        **content,
        "statement_digest": digest,
        "created_at": created_at,
    }


# -------------------------------------------------------------- idempotency


def _replay_or_conflict(statement_id: str, existing: SettlementStatement,
                        request_digest: str,
                        cycles: list[dict]) -> StatementResult:
    if existing.request_digest == request_digest:
        return StatementResult(json.loads(existing.statement_json), True)
    raise _statement_conflict(
        statement_id, existing, request_digest, cycles)


def _classify_integrity_error(eng, statement_id, cycles, request_digest, exc):
    with Session(eng) as session:
        row = session.get(SettlementStatement, statement_id)
        if row is not None:
            return _replay_or_conflict(
                statement_id, row, request_digest, cycles)
    raise exc


# --------------------------------------------------------------- read model


def get_statement(statement_id: str, *, engine=None) -> dict:
    """返回不可变综合声明（响应体与首次生成逐字节一致）。

    声明不存在 → 404 ``statement_not_found``。
    """
    eng = engine or default_engine
    with Session(eng) as session:
        row = session.get(SettlementStatement, statement_id)
        if row is None:
            raise PricingError(
                f"综合声明不存在: {statement_id}",
                code="statement_not_found", status_code=404,
                details={"statement_id": statement_id})
        return json.loads(row.statement_json)
