"""
NarrateKPI — FastAPI Server (Module 3)

REST API for the Human-in-the-Loop Review Queue.  Triggers the Math Engine
+ LLM Synthesis pipeline, stores reports in a JSON file, and serves a
dark-mode SPA for agency managers to review, edit, and approve reports.

Endpoints
---------
POST   /api/reports/generate-all   — Run pipeline for all mock clients
GET    /api/reports                 — List all reports (filterable by status)
GET    /api/reports/{id}            — Get single report with full details
PUT    /api/reports/{id}            — Update markdown content → sets IN_REVIEW
POST   /api/reports/{id}/approve    — Approve report → sets APPROVED
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure sibling modules are importable when run directly.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from email_service import send_report_email
from llm_engine import generate_report
from math_engine import run_math_engine
from mock_data import ALL_CASES
from schemas import MetricSet, WeeklyComparisonInput
from storage import (
    ReportRecord,
    ReportStatus,
    _now_iso,
    clear_all,
    load_all,
    load_by_id,
    save,
)

# ──────────────────────────────────────────────────────────────────────
#  App setup
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NarrateKPI",
    description="AI-Driven Agency Report Automation — Review Queue API",
    version="3.0.0",
    redirect_slashes=True,
)

# Track server start time for uptime reporting
_START_TIME = time.time()

# ── CORS: same-origin SPA doesn't need it, but Render's proxy may
#     require headers for asset preflight checks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = HERE / "static"

# ── Auto-create runtime data directories ─────────────────────────────
_RUNTIME_DIRS = [
    HERE / "data",
    HERE / "reports_output",
    HERE / "email_output",
]
for _dir in _RUNTIME_DIRS:
    _dir.mkdir(parents=True, exist_ok=True)
print(f"[NarrateKPI] 📁 Ensured {len(_RUNTIME_DIRS)} runtime directories exist")


# ──────────────────────────────────────────────────────────────────────
#  Request / Response schemas
# ──────────────────────────────────────────────────────────────────────

class ReportSummary(BaseModel):
    """Lightweight summary returned in the list endpoint."""

    id: str
    account_id: str
    client_name: str
    period: str
    status: str
    anomalies_found: int
    created_at: str
    updated_at: str


class ReportDetail(BaseModel):
    """Full report returned by the detail endpoint."""

    id: str
    account_id: str
    client_name: str
    period: str
    status: str
    raw_metrics: dict
    anomalies_found: int
    markdown_content: str
    target_email: Optional[str] = None
    sent_at: Optional[str] = None
    created_at: str
    updated_at: str


class UpdateReportRequest(BaseModel):
    """Payload for editing a report's Markdown content."""

    markdown_content: str = Field(..., min_length=1, description="Updated Markdown report text")


class GenerateAllResponse(BaseModel):
    """Response returned after generating reports for all mock clients."""

    generated: int
    reports: List[ReportSummary]


class CustomReportRequest(BaseModel):
    """Payload for creating a report with custom client metrics."""

    client_name: str = Field(..., min_length=1, description="Client/account name")
    period: str = Field(..., min_length=1, description="ISO week period, e.g. 2026-W30")
    target_email: Optional[str] = Field(default=None, description="Email to send the report to")
    impressions: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    spend: float = Field(..., ge=0.0)
    conversions: int = Field(..., ge=0)
    revenue: float = Field(..., ge=0.0)
    prev_impressions: int = Field(default=0, ge=0, description="Previous period impressions")
    prev_clicks: int = Field(default=0, ge=0, description="Previous period clicks")
    prev_spend: float = Field(default=0.0, ge=0.0, description="Previous period spend")
    prev_conversions: int = Field(default=0, ge=0, description="Previous period conversions")
    prev_revenue: float = Field(default=0.0, ge=0.0, description="Previous period revenue")


class SendReportResponse(BaseModel):
    """Response returned after sending a report via email."""

    id: str
    status: str
    sent_at: str
    sent_to: str


# ──────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check() -> dict:
    """Health-check endpoint for monitoring and Docker HEALTHCHECK.

    Returns app version, uptime, store path, runtime directories, and
    the status of optional LLM / Email providers so operators can
    quickly assess the instance's capabilities.
    """
    from storage import STORE_PATH
    import email_service as es
    import llm_engine as llm

    uptime = time.time() - _START_TIME

    # Check runtime directories
    dirs_status = {}
    for _dir in _RUNTIME_DIRS:
        dirs_status[str(_dir.name)] = {
            "exists": _dir.is_dir(),
            "writable": os.access(str(_dir), os.W_OK) if _dir.is_dir() else False,
        }

    # Check store writability
    store_path = STORE_PATH
    store_parent = Path(store_path).parent
    store_writable = store_parent.is_dir() and os.access(str(store_parent), os.W_OK)

    # Detect LLM provider
    provider = llm.detect_provider()
    llm_status = "dry_run"
    llm_detail = "No API key configured — reports generated in dry-run mode"
    if provider:
        llm_status = "live"
        llm_detail = f"{provider[0]} ({provider[2]})"

    # Email provider
    email_status = "dry_run"
    email_detail = "RESEND_API_KEY not set — emails logged to disk"
    if es.RESEND_API_KEY:
        email_status = "live"
        email_detail = f"Resend ({es.DEFAULT_FROM_EMAIL})"

    return {
        "status": "ok",
        "version": "3.0.0",
        "uptime_seconds": round(uptime, 1),
        "uptime_human": _format_uptime(uptime),
        "store": {
            "path": store_path,
            "writable": store_writable,
        },
        "directories": dirs_status,
        "providers": {
            "llm": {"status": llm_status, "detail": llm_detail},
            "email": {"status": email_status, "detail": email_detail},
        },
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> str:
    """Serve the public landing page."""
    landing_path = _static_dir / "landing.html"
    if landing_path.is_file():
        return landing_path.read_text(encoding="utf-8")
    return HTMLResponse(
        content="<h1>NarrateKPI</h1><p>Landing page not found. Ensure static/landing.html exists.</p>",
        status_code=200,
    )


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def app_spa() -> str:
    """Serve the Review Queue SPA at /app."""
    index_path = _static_dir / "index.html"
    if index_path.is_file():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse(
        content="<h1>NarrateKPI</h1><p>Static index not found. Run with --reload after creating static/index.html</p>",
        status_code=200,
    )


@app.post("/api/reports/generate-all", response_model=GenerateAllResponse)
async def generate_all() -> GenerateAllResponse:
    """Run the Math Engine + LLM Synthesis for every mock client.

    Clears any existing reports and regenerates from scratch.
    """
    try:
        clear_all()
        created: List[ReportRecord] = []

        for case in ALL_CASES:
            # ── Step 1: Math Engine ────────────────────────────────
            anomaly_report = run_math_engine(case)
            summary = anomaly_report.summary_raw_json

            # ── Step 2: LLM Synthesis (or dry-run mock) ────────────
            markdown = generate_report(summary)

            # ── Step 3: Persist ────────────────────────────────────
            record = ReportRecord(
                account_id=summary.get("account_id", case.account_id),
                client_name=summary.get("client_name", case.client_name),
                period=summary.get("period_current", ""),
                status=ReportStatus.DRAFT,
                raw_metrics={
                    "current": summary.get("current_metrics", {}),
                    "previous": summary.get("previous_metrics", {}),
                },
                anomalies_found=summary.get("total_anomalies_found", 0),
                markdown_content=markdown,
            )
            save(record)
            created.append(record)

        return GenerateAllResponse(
            generated=len(created),
            reports=[_to_summary(r) for r in created],
        )
    except Exception as e:
        print(f"[NarrateKPI] ❌ generate_all() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"generate_all: {e}")


@app.get("/api/reports", response_model=List[ReportSummary])
async def list_reports(
    status: Optional[str] = Query(None, description="Filter by status (DRAFT, IN_REVIEW, APPROVED, SENT)"),
) -> List[ReportSummary]:
    """List all reports, optionally filtered by status."""
    try:
        records = load_all()
        if status:
            status_upper = status.upper()
            records = [r for r in records if r.status.value == status_upper]
        # Most recent first.
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return [_to_summary(r) for r in records]
    except Exception as e:
        print(f"[NarrateKPI] ❌ list_reports() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"list_reports: {e}")


@app.get("/api/reports/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str) -> ReportDetail:
    """Get a single report with full Markdown content and raw metrics."""
    try:
        record = load_by_id(report_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Report not found")
        return _to_detail(record)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NarrateKPI] ❌ get_report() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"get_report: {e}")


@app.put("/api/reports/{report_id}", response_model=ReportDetail)
async def update_report(report_id: str, body: UpdateReportRequest) -> ReportDetail:
    """Update the Markdown content of a report.

    Automatically transitions status to ``IN_REVIEW`` if it is currently
    ``DRAFT`` or ``IN_REVIEW``.
    """
    try:
        record = load_by_id(report_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Report not found")

        # Only DRAFT and IN_REVIEW reports can be edited; approved/sent are locked.
        if record.status not in (ReportStatus.DRAFT, ReportStatus.IN_REVIEW):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot edit a report with status '{record.status.value}'. Only DRAFT and IN_REVIEW reports can be modified.",
            )
        record.markdown_content = body.markdown_content
        record.status = ReportStatus.IN_REVIEW
        save(record)
        return _to_detail(record)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NarrateKPI] ❌ update_report() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"update_report: {e}")


@app.post("/api/reports/{report_id}/approve", response_model=ReportDetail)
async def approve_report(report_id: str) -> ReportDetail:
    """Approve a report, transitioning it to ``APPROVED``.

    Only ``DRAFT`` or ``IN_REVIEW`` reports can be approved.
    """
    try:
        record = load_by_id(report_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Report not found")
        if record.status in (ReportStatus.APPROVED, ReportStatus.SENT):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot approve a report with status '{record.status.value}'",
            )
        record.status = ReportStatus.APPROVED
        save(record)
        return _to_detail(record)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NarrateKPI] ❌ approve_report() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"approve_report: {e}")


# ──────────────────────────────────────────────────────────────────────
#  Module 4: Custom Data Ingestion & Email Dispatch
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/clients/custom-report", response_model=ReportDetail)
async def create_custom_report(body: CustomReportRequest) -> ReportDetail:
    """Accept custom client metrics, run the full pipeline, and create a
    ``DRAFT`` report record.

    The request provides current-period metrics and (optionally) previous-period
    metrics.  The Math Engine auto-calculates derived fields (CTR, CPC, CPA,
    ROAS) and detects anomalies.  The LLM Synthesis engine generates the
    narrative report.
    """
    try:
        current = MetricSet(
            impressions=body.impressions,
            clicks=body.clicks,
            spend=body.spend,
            conversions=body.conversions,
            revenue=body.revenue,
        )
        previous = MetricSet(
            impressions=body.prev_impressions,
            clicks=body.prev_clicks,
            spend=body.prev_spend,
            conversions=body.prev_conversions,
            revenue=body.prev_revenue,
        )

        comparison = WeeklyComparisonInput(
            account_id=f"custom_{uuid.uuid4().hex[:6]}",
            client_name=body.client_name,
            period_current=body.period,
            period_previous=_derive_previous_period(body.period),
            current_metrics=current,
            previous_metrics=previous,
        )

        # ── Step 1: Math Engine ────────────────────────────────────
        anomaly_report = run_math_engine(comparison)
        summary = anomaly_report.summary_raw_json

        # ── Step 2: LLM Synthesis ──────────────────────────────────
        markdown = generate_report(summary)

        # ── Step 3: Persist ────────────────────────────────────────
        record = ReportRecord(
            account_id=comparison.account_id,
            client_name=body.client_name,
            period=body.period,
            status=ReportStatus.DRAFT,
            raw_metrics={
                "current": summary.get("current_metrics", {}),
                "previous": summary.get("previous_metrics", {}),
            },
            anomalies_found=summary.get("total_anomalies_found", 0),
            markdown_content=markdown,
            target_email=body.target_email,
        )
        save(record)
        return _to_detail(record)
    except Exception as e:
        print(f"[NarrateKPI] ❌ create_custom_report() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"create_custom_report: {e}")


@app.post("/api/reports/{report_id}/send", response_model=SendReportResponse)
async def send_report(report_id: str) -> SendReportResponse:
    """Send an approved report via email.

    Converts the report Markdown to email-safe HTML and dispatches it via
    the Resend API (or dry-run logging if no API key is configured).
    Only ``APPROVED`` reports can be sent.
    """
    try:
        record = load_by_id(report_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Report not found")

        if record.status != ReportStatus.APPROVED:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot send a report with status '{record.status.value}'. Only APPROVED reports can be sent.",
            )

        target = record.target_email or "client@example.com"
        subject = f"Weekly Performance Report — {record.client_name} ({record.period})"
        success = send_report_email(
            to_email=target,
            subject=subject,
            markdown_content=record.markdown_content,
        )

        now = _now_iso()
        record.status = ReportStatus.SENT
        record.sent_at = now
        save(record)

        return SendReportResponse(
            id=record.report_id,
            status=record.status.value,
            sent_at=now,
            sent_to=target,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NarrateKPI] ❌ send_report() failed:\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"send_report: {e}")


# ──────────────────────────────────────────────────────────────────────
#  Static file serving (registered AFTER API routes to avoid shadowing)
# ──────────────────────────────────────────────────────────────────────

if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

import uuid  # noqa: E402 (import after endpoints for readability)


def _format_uptime(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Examples:
        ``_format_uptime(45)`` → ``"45s"``
        ``_format_uptime(125)`` → ``"2m 5s"``
        ``_format_uptime(3661)`` → ``"1h 1m 1s"``
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _derive_previous_period(current: str) -> str:
    """Derive the previous ISO week from a current week string.

    E.g. ``"2026-W30"`` → ``"2026-W29"``.  Handles year boundaries
    (``"2026-W01"`` → ``"2025-W52"``).
    """
    from datetime import datetime, timedelta
    # Parse the ISO week into a Monday date, subtract 7 days, format back.
    year = int(current[:4])
    week = int(current[6:])
    # ISO week 1 of year Y starts on the Monday of the week containing Jan 4.
    jan4 = datetime(year, 1, 4)
    # Monday of week 1
    monday_w1 = jan4 - timedelta(days=jan4.weekday())
    target_monday = monday_w1 + timedelta(weeks=week - 2)
    prev_year = target_monday.isocalendar()[0]
    prev_week = target_monday.isocalendar()[1]
    return f"{prev_year}-W{prev_week:02d}"


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def _to_summary(record: ReportRecord) -> ReportSummary:
    return ReportSummary(
        id=record.report_id,
        account_id=record.account_id,
        client_name=record.client_name,
        period=record.period,
        status=record.status.value,
        anomalies_found=record.anomalies_found,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_detail(record: ReportRecord) -> ReportDetail:
    return ReportDetail(
        id=record.report_id,
        account_id=record.account_id,
        client_name=record.client_name,
        period=record.period,
        status=record.status.value,
        raw_metrics=record.raw_metrics,
        anomalies_found=record.anomalies_found,
        markdown_content=record.markdown_content,
        target_email=record.target_email,
        sent_at=record.sent_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
