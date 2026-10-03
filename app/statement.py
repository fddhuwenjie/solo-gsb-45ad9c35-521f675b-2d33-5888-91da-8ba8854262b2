"""多个已关账账期的综合结算与碳排声明。

调用方提交 ``statement_id``、1～12 个 ``cycle_id`` 及每个账期期望的
``detail_digest``（关账快照摘要）与 ``report_digest``（碳排报告摘要），
声明在**单个 SQLite 写事务**内完成全部核对与落库：

1. 先按 ``statement_id`` 判幂等：同 ID 同内容（账期列表 + 两个摘要，
   按 cycle_id 排序后取 SHA-256，请求顺序不影响判定）重放原声明
   （HTTP 200），换账期列表或换摘要 → 409 ``statement_conflict``；
2. 逐账期重读 ``CycleClosure`` 与 ``CarbonReport``：账期不存在 → 404
   ``cycle_not_found``；未关账 → 409 ``cycle_not_closed``；没有碳排
   报告 → 409 ``carbon_report_missing``；期望摘要与持久化摘要不符 →
   409 ``detail_digest_mismatch`` / ``report_digest_mismatch``；报告
   revision 与关账 revision 不一致 → 409 ``revision_mismatch``；
3. 跨账期核对：规则版本的币种与能量单位必须一致（409
   ``incompatible_cycles``）；各账期关账快照中的 UTC 用量区间不得重叠
   （首尾相接允许，409 ``cycle_interval_overlap``）；
4. 任一缺失、摘要不符、重复账期（400 ``duplicate_cycle``）或状态不合
   即整体拒绝——整笔事务回滚，不落任何数据；
5. 全部通过后按**关账时间**（同刻按 cycle_id 字典序）输出逐账期总用量、
   总金额、总排放与碳强度（排放量/用量，六位小数），并给出三者各自的
   合计与综合碳强度。**金额、用量、排放分别守恒**：合计直接取自关账
   快照与碳排报告的持久化数值之和，绝不从实时批次重算；
6. 声明以规范 JSON 的 SHA-256（``statement_digest``，不覆盖
   ``created_at``）落库且不可变，重启后凭 ``statement_id`` 重放或查询。
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

from .database import (
    BillingCycle,
    CarbonReport,
    CycleClosure,
    Statement,
    engine as default_engine,
)
from .importer import canonical_json
from .pricing import Q2, Q6, ZERO, PricingError, load_version
from .schemas import StatementRequest

STATEMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
EMISSION_UNIT = "kgCO2e"

# 与批次提交/关账/碳排相同的写锁竞争重试策略：延迟事务升级写锁撞上
# SQLITE_BUSY 时，在全新事务里重读状态后整笔重跑。
_MAX_LOCK_RETRIES = 20


@dataclass(frozen=True)
class StatementResult:
    body: dict
    replayed: bool


def _utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_statement_id(statement_id: str) -> None:
    if not STATEMENT_ID_RE.fullmatch(statement_id):
        raise PricingError(
            f"非法 statement_id: {statement_id!r}；只允许字母/数字开头，"
            "后跟至多 127 个字母、数字、下划线或连字符",
            code="invalid_statement_id", status_code=400,
            details={"statement_id": statement_id})


# --------------------------------------------------------------------- errors


def _invalid_digest(field: str, cycle_id: str, value: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 {field} 不是 64 位小写十六进制摘要: {value!r}",
        code="invalid_digest", status_code=400,
        details={"cycle_id": cycle_id, "field": field, "value": value})


def _duplicate_cycle(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 在请求中出现多次；一份声明内账期必须唯一",
        code="duplicate_cycle", status_code=400,
        details={"cycle_id": cycle_id})


def _cycle_not_found(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期不存在: {cycle_id}",
        code="cycle_not_found", status_code=404,
        details={"cycle_id": cycle_id})


def _cycle_not_closed(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 尚未关账，不能并入综合声明",
        code="cycle_not_closed", status_code=409,
        details={"cycle_id": cycle_id})


def _carbon_report_missing(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 还没有碳排报告，不能并入综合声明",
        code="carbon_report_missing", status_code=409,
        details={"cycle_id": cycle_id})


def _detail_digest_mismatch(cycle_id: str, expected: str,
                            actual: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的期望 detail_digest 与关账快照不一致："
        f"期望 {expected}，实际 {actual}，拒绝生成声明",
        code="detail_digest_mismatch", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_detail_digest": expected,
                 "actual_detail_digest": actual})


def _report_digest_mismatch(cycle_id: str, expected: str,
                            actual: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的期望 report_digest 与碳排报告不一致："
        f"期望 {expected}，实际 {actual}，拒绝生成声明",
        code="report_digest_mismatch", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_report_digest": expected,
                 "actual_report_digest": actual})


def _revision_mismatch(cycle_id: str, closure_revision: int,
                       report_revision: int) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的碳排报告 revision（{report_revision}）与关账 "
        f"revision（{closure_revision}）不一致：报告不是针对关账状态生成的，"
        "请按关账 revision 重新生成碳排报告",
        code="revision_mismatch", status_code=409,
        details={"cycle_id": cycle_id,
                 "closure_revision": closure_revision,
                 "report_revision": report_revision})


def _incompatible_cycles(field: str, values: dict[str, str]) -> PricingError:
    label = "币种" if field == "currency" else "能量单位"
    return PricingError(
        f"各账期规则版本的{label}不一致（{values}），不能并入同一份综合声明",
        code="incompatible_cycles", status_code=409,
        details={"field": field, "values": values})


def _cycle_interval_overlap(a: dict, b: dict) -> PricingError:
    return PricingError(
        f"账期 {a['cycle_id']} 的区间 {a['start']} ~ {a['end']} 与账期 "
        f"{b['cycle_id']} 的区间 {b['start']} ~ {b['end']} 在 UTC 时间线上"
        "重叠；并入同一份声明的各账期间隔必须互不重叠（首尾相接允许）",
        code="cycle_interval_overlap", status_code=409,
        details={"interval_a": a, "interval_b": b})


def _statement_conflict(statement_id: str, existing: Statement,
                        received_digest: str) -> PricingError:
    return PricingError(
        f"statement_id {statement_id!r} 已用于内容不同的声明（首次覆盖 "
        f"{existing.cycle_count} 个账期）；同 statement_id 仅允许同内容重放",
        code="statement_conflict", status_code=409,
        details={"statement_id": statement_id,
                 "existing": {
                     "cycle_count": existing.cycle_count,
                     "content_digest": existing.content_digest,
                     "statement_digest": existing.statement_digest,
                 },
                 "received": {"content_digest": received_digest}})


# --------------------------------------------------------------------- digest


def _content_digest(req: StatementRequest) -> str:
    """幂等内容摘要：账期 + 两个期望摘要（按 cycle_id 排序）。

    请求中账期的罗列顺序不影响摘要；``statement_id`` 是幂等键本身，
    不参与摘要。
    """
    doc = {
        "cycles": [
            {"cycle_id": c.cycle_id,
             "detail_digest": c.detail_digest,
             "report_digest": c.report_digest}
            for c in sorted(req.cycles, key=lambda c: c.cycle_id)
        ],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def _validate_request(req: StatementRequest) -> None:
    """事务外的请求自检：ID 格式、摘要格式与账期唯一性。"""
    validate_statement_id(req.statement_id)
    seen: set[str] = set()
    for c in req.cycles:
        if not DIGEST_RE.fullmatch(c.detail_digest):
            raise _invalid_digest("detail_digest", c.cycle_id, c.detail_digest)
        if not DIGEST_RE.fullmatch(c.report_digest):
            raise _invalid_digest("report_digest", c.cycle_id, c.report_digest)
        if c.cycle_id in seen:
            raise _duplicate_cycle(c.cycle_id)
        seen.add(c.cycle_id)


# -------------------------------------------------------------- idempotency


def _replay_or_conflict(existing: Statement, statement_id: str,
                        digest: str) -> StatementResult:
    if existing.content_digest == digest:
        return StatementResult(json.loads(existing.statement_json), True)
    raise _statement_conflict(statement_id, existing, digest)


# ------------------------------------------------------------ per-cycle load


def _load_cycle(session: Session, req_cycle) -> dict:
    """事务内重读一个账期的关账快照与碳排报告并核对期望摘要。

    返回按声明输出组织的逐账期行（数值全部取自持久化的快照/报告，
    不从实时批次重算）。
    """
    cycle_id = req_cycle.cycle_id
    cycle = session.get(BillingCycle, cycle_id)
    if cycle is None:
        raise _cycle_not_found(cycle_id)
    closure = session.scalar(
        select(CycleClosure).where(CycleClosure.cycle_id == cycle_id))
    if closure is None or cycle.status != "closed":
        raise _cycle_not_closed(cycle_id)
    report = session.get(CarbonReport, cycle_id)
    if report is None:
        raise _carbon_report_missing(cycle_id)
    if closure.detail_digest != req_cycle.detail_digest:
        raise _detail_digest_mismatch(
            cycle_id, req_cycle.detail_digest, closure.detail_digest)
    if report.report_digest != req_cycle.report_digest:
        raise _report_digest_mismatch(
            cycle_id, req_cycle.report_digest, report.report_digest)
    if report.revision != closure.revision:
        raise _revision_mismatch(cycle_id, closure.revision, report.revision)

    version = load_version(session, cycle.version_code)
    energy = Decimal(closure.total_energy)
    emissions = Decimal(report.total_emissions)
    intensity = ((emissions / energy).quantize(Q6, rounding=ROUND_HALF_UP)
                 if energy > 0 else ZERO.quantize(Q6))
    snap = json.loads(closure.snapshot_json)
    return {
        "row": {
            "cycle_id": cycle_id,
            "version": cycle.version_code,
            "revision": closure.revision,
            "close_id": closure.close_id,
            "closed_at": closure.created_at,
            "detail_digest": closure.detail_digest,
            "report_digest": report.report_digest,
            "factor_version": report.factor_version,
            "total_energy": closure.total_energy,
            "total_amount": closure.total_amount,
            "total_emissions": report.total_emissions,
            "carbon_intensity": str(intensity),
        },
        "currency": version.currency,
        "energy_unit": version.energy_unit,
        # 不可变关账快照中的 UTC 区间（用于跨账期重叠核对）。
        "intervals": [
            {"cycle_id": cycle_id, "start": i["start"], "end": i["end"]}
            for i in snap["intervals"]
        ],
    }


def _check_compatible(loaded: list[dict]) -> None:
    """全部账期的币种与能量单位必须一致。"""
    for field in ("currency", "energy_unit"):
        values = {item["row"]["cycle_id"]: item[field] for item in loaded}
        if len(set(values.values())) > 1:
            raise _incompatible_cycles(field, values)


def _check_no_overlap(loaded: list[dict]) -> None:
    """各账期关账快照中的 UTC 区间互不重叠（首尾相接允许）。"""
    spans = sorted(
        (span for item in loaded for span in item["intervals"]),
        key=lambda s: (_parse_z(s["start"]), _parse_z(s["end"])),
    )
    for k in range(1, len(spans)):
        a, b = spans[k - 1], spans[k]
        if _parse_z(b["start"]) < _parse_z(a["end"]):
            raise _cycle_interval_overlap(a, b)


# ----------------------------------------------------------------- main flow


def create_statement(req: StatementRequest, *, engine=None) -> StatementResult:
    """生成跨账期综合声明，返回 ``(声明体, 是否幂等重放)``。

    任何校验失败都抛出 :class:`PricingError` 且整笔事务回滚——不写入
    声明。``engine`` 可注入（测试用临时文件库），默认应用全局引擎。
    """
    _validate_request(req)
    digest = _content_digest(req)
    eng = engine or default_engine

    attempt = 0
    while True:
        attempt += 1
        try:
            return _create_once(eng, req, digest)
        except OperationalError as exc:  # SQLITE_BUSY write-lock race
            if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                raise
            # 全新事务重读声明/账期状态后整笔重跑。


def _is_lock_error(exc: OperationalError) -> bool:
    msg = str(exc.orig).lower() if exc.orig is not None else ""
    return ("database is locked" in msg
            or "database table is locked" in msg)


def _create_once(eng, req: StatementRequest, digest: str) -> StatementResult:
    with Session(eng) as session:
        try:
            with session.begin():
                # 1) 幂等优先：同 statement_id 同内容重放、异内容冲突，
                #    先于一切账期状态判定。
                existing = session.get(Statement, req.statement_id)
                if existing is not None:
                    return _replay_or_conflict(existing, req.statement_id,
                                               digest)

                # 2) 同一事务内逐账期重读关账快照与碳排报告并核对摘要。
                loaded = [_load_cycle(session, c) for c in req.cycles]

                # 3) 跨账期核对：币种/能量单位一致、UTC 区间不重叠。
                _check_compatible(loaded)
                _check_no_overlap(loaded)

                # 4) 按关账时间（同刻按 cycle_id）排序；合计取持久化数值
                #    之和——金额、用量、排放分别守恒，不从实时批次重算。
                loaded.sort(key=lambda item: (_parse_z(
                    item["row"]["closed_at"]), item["row"]["cycle_id"]))
                rows = [item["row"] for item in loaded]
                total_energy = sum(
                    (Decimal(r["total_energy"]) for r in rows), ZERO)
                total_amount = sum(
                    (Decimal(r["total_amount"]) for r in rows), ZERO
                ).quantize(Q2, rounding=ROUND_HALF_UP)
                total_emissions = sum(
                    (Decimal(r["total_emissions"]) for r in rows), ZERO
                ).quantize(Q6)
                intensity = (
                    (total_emissions / total_energy).quantize(
                        Q6, rounding=ROUND_HALF_UP)
                    if total_energy > 0 else ZERO.quantize(Q6))

                content = {
                    "statement_id": req.statement_id,
                    "cycle_count": len(rows),
                    "currency": loaded[0]["currency"],
                    "energy_unit": loaded[0]["energy_unit"],
                    "emission_unit": EMISSION_UNIT,
                    "cycles": rows,
                    "total_energy": str(total_energy),
                    "total_amount": str(total_amount),
                    "total_emissions": str(total_emissions),
                    "carbon_intensity": str(intensity),
                }
                # 声明摘要只覆盖确定性内容（created_at 不参与），供离线核对。
                statement_digest = hashlib.sha256(
                    canonical_json(content).encode("utf-8")).hexdigest()
                created_at = _now()
                body = {**content, "statement_digest": statement_digest,
                        "created_at": created_at}

                session.add(Statement(
                    statement_id=req.statement_id,
                    cycle_count=len(rows),
                    content_digest=digest,
                    statement_digest=statement_digest,
                    currency=loaded[0]["currency"],
                    energy_unit=loaded[0]["energy_unit"],
                    total_energy=str(total_energy),
                    total_amount=str(total_amount),
                    total_emissions=str(total_emissions),
                    statement_json=json.dumps(body, ensure_ascii=False),
                    created_at=created_at,
                ))
                return StatementResult(body, False)

        except IntegrityError as exc:
            # 主键冲突（并发同 statement_id 生成）：只读重查归类为重放 /
            # 冲突，绝不写任何数据。
            return _classify_integrity_error(eng, req, digest, exc)


def _classify_integrity_error(eng, req, digest, exc):
    with Session(eng) as session:
        row = session.get(Statement, req.statement_id)
        if row is not None:
            return _replay_or_conflict(row, req.statement_id, digest)
    raise exc


# --------------------------------------------------------------- read model


def get_statement(statement_id: str, *, engine=None) -> dict:
    """返回已保存的不可变综合声明；不存在 → 404 ``statement_not_found``。"""
    validate_statement_id(statement_id)
    eng = engine or default_engine
    with Session(eng) as session:
        row = session.get(Statement, statement_id)
        if row is None:
            raise PricingError(
                f"声明不存在: {statement_id}",
                code="statement_not_found", status_code=404,
                details={"statement_id": statement_id})
        return json.loads(row.statement_json)
