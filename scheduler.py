"""
NarrateKPI — Weekly Report Automation Scheduler (Module 7)

APScheduler-based background task that runs every Monday at 09:00 to
automatically generate KPI reports for all active clients in the database.
Also provides lifecycle management hooks (``start_scheduler`` /
``stop_scheduler``) and a status query function for the API endpoints.

Design decisions
----------------
- Uses ``BackgroundScheduler`` (thread-pool based) rather than
  ``AsyncIOScheduler`` so the scheduled job doesn't compete with the
  FastAPI event loop.
- Creates its own DB session via ``database.SessionLocal()`` — it does
  **not** depend on the ``get_db()`` FastAPI dependency.
- Each client is processed inside a ``try/except`` block so a failure on
  one client never blocks report generation for others.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure sibling modules are importable when the scheduler runs.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import email_service as es
from database import Client, Report, ReportStatus, SessionLocal, _now_iso
from llm_engine import generate_report
from math_engine import run_math_engine
from schemas import MetricSet, WeeklyComparisonInput


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

SCHEDULER_TIMEZONE: str = os.environ.get("SCHEDULER_TIMEZONE", "Europe/Kyiv")

# ── Module-level state ───────────────────────────────────────────────
_scheduler: Optional[BackgroundScheduler] = None
_start_time: Optional[datetime] = None


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────


def _current_iso_week() -> str:
    """Return the current ISO week string, e.g. ``"2026-W30"``."""
    today = datetime.now(timezone.utc)
    iso = today.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _previous_iso_week() -> str:
    """Return the previous ISO week string."""
    last_week = datetime.now(timezone.utc) - timedelta(days=7)
    iso = last_week.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _generate_mock_case_for_client(
    client: Client,
    period_current: str,
    period_previous: str,
) -> WeeklyComparisonInput:
    """Generate a plausible ``WeeklyComparisonInput`` for a single client.

    Uses a deterministic hash of ``account_id`` so each client gets
    consistent but varied mock metrics across runs.
    """
    seed = sum(ord(c) for c in client.account_id) % 100

    base_impressions = 20_000 + seed * 2_000
    base_clicks = int(base_impressions * (0.02 + (seed % 10) * 0.003))  # 2–5 % CTR

    current = MetricSet(
        impressions=max(1000, int(base_impressions * (1.0 + (seed % 5 - 2) * 0.05))),
        clicks=max(50, int(base_clicks * (1.0 + (seed % 7 - 3) * 0.05))),
        spend=max(100.0, round(500.0 + seed * 150.0, 2)),
        conversions=max(5, int(20 + seed * 2)),
        revenue=max(200.0, round(2000.0 + seed * 400.0, 2)),
    )

    previous = MetricSet(
        impressions=max(1000, int(base_impressions * (0.9 + (seed % 3) * 0.05))),
        clicks=max(50, int(base_clicks * (0.85 + (seed % 4) * 0.05))),
        spend=max(100.0, round(450.0 + seed * 140.0, 2)),
        conversions=max(5, int(18 + seed * 2)),
        revenue=max(200.0, round(1800.0 + seed * 380.0, 2)),
    )

    return WeeklyComparisonInput(
        account_id=client.account_id,
        client_name=client.client_name,
        period_current=period_current,
        period_previous=period_previous,
        current_metrics=current,
        previous_metrics=previous,
    )


# ──────────────────────────────────────────────────────────────────────
#  Core scheduled task
# ──────────────────────────────────────────────────────────────────────


def scheduled_generate_all_reports() -> Dict[str, Any]:
    """Generate weekly reports for **every** client in the database.

    Called by the APScheduler cron trigger (Monday 09:00) or manually
    via ``POST /api/scheduler/trigger-now``.

    Each client is wrapped in a ``try/except`` so a single failure never
    blocks the rest of the batch.

    Returns
    -------
    dict
        Summary with counts of successes, failures, and emails sent.
    """
    period_current = _current_iso_week()
    period_previous = _previous_iso_week()

    print(
        f"[NarrateKPI] ⏰ Scheduled report generation started: "
        f"{period_current} (prev {period_previous})",
        flush=True,
    )

    db = SessionLocal()
    try:
        clients = db.query(Client).all()
        print(
            f"[NarrateKPI] 📋 Found {len(clients)} client(s) in database",
            flush=True,
        )

        results: Dict[str, Any] = {
            "period_current": period_current,
            "period_previous": period_previous,
            "total_clients": len(clients),
            "success": 0,
            "failed": 0,
            "emails_sent": 0,
            "errors": [],
        }

        for client in clients:
            try:
                _process_single_client(
                    db=db,
                    client=client,
                    period_current=period_current,
                    period_previous=period_previous,
                    results=results,
                )
            except Exception as exc:
                results["failed"] += 1
                results["errors"].append(
                    {"client": client.client_name, "error": str(exc)[:300]}
                )
                print(
                    f"[NarrateKPI] ❌ Failed for {client.client_name}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                db.rollback()

        db.commit()

        summary = (
            f"{results['success']} success, {results['failed']} failed, "
            f"{results['emails_sent']} emails sent"
        )
        print(f"[NarrateKPI] ✅ Scheduled run complete: {summary}", flush=True)
        return results

    except Exception as exc:
        print(
            f"[NarrateKPI] ❌ Scheduled run failed catastrophically: {exc}",
            flush=True,
        )
        traceback.print_exc()
        db.rollback()
        return {
            "period_current": period_current,
            "error": str(exc)[:500],
        }
    finally:
        db.close()


def _process_single_client(
    db: SessionLocal,
    client: Client,
    period_current: str,
    period_previous: str,
    results: Dict[str, Any],
) -> None:
    """Run the full pipeline (math → LLM → persist → email) for one client."""
    from uuid import uuid4

    print(f"[NarrateKPI] 📊 {client.client_name} — generating...", flush=True)

    # ── Step 1: Mock comparison case ──────────────────────────────
    case = _generate_mock_case_for_client(
        client=client,
        period_current=period_current,
        period_previous=period_previous,
    )

    # ── Step 2: Math Engine ────────────────────────────────────────
    anomaly_report = run_math_engine(case)
    summary = anomaly_report.summary_raw_json

    # ── Step 3: LLM Synthesis (or dry-run mock) ────────────────────
    markdown = generate_report(summary)

    # ── Step 4: Persist as IN_REVIEW (scheduled = pre-reviewed) ────
    now = _now_iso()
    report = Report(
        report_id=uuid4().hex[:12],
        client_id=client.id,
        account_id=client.account_id,
        client_name=client.client_name,
        period=period_current,
        status=ReportStatus.IN_REVIEW.value,
        raw_metrics={
            "current": summary.get("current_metrics", {}),
            "previous": summary.get("previous_metrics", {}),
        },
        anomalies_found=summary.get("total_anomalies_found", 0),
        markdown_content=markdown,
        target_email=client.target_email,
        created_at=now,
        updated_at=now,
    )
    db.add(report)
    db.flush()

    results["success"] += 1
    print(
        f"[NarrateKPI] ✅ {client.client_name} — report saved ({report.report_id})",
        flush=True,
    )

    # ── Step 5: Email notification (if configured) ─────────────────
    _maybe_send_email(client, period_current, markdown, results)


def _maybe_send_email(
    client: Client,
    period: str,
    markdown: str,
    results: Dict[str, Any],
) -> None:
    """Send an email notification if the client has a target address
    and an email provider is configured."""
    target = client.target_email
    if not target:
        return

    subject = f"Weekly Performance Report — {client.client_name} ({period})"

    try:
        success = es.send_report_email(
            to_email=target,
            subject=subject,
            markdown_content=markdown,
        )
        if success:
            results["emails_sent"] += 1
            print(
                f"[NarrateKPI] 📧 Email sent to {target} for {client.client_name}",
                flush=True,
            )
        else:
            print(
                f"[NarrateKPI] ⚠️  Email delivery failed for "
                f"{client.client_name} ({target})",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[NarrateKPI] ⚠️  Email exception for {client.client_name}: {exc}",
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────
#  Scheduler lifecycle management
# ──────────────────────────────────────────────────────────────────────


def start_scheduler() -> None:
    """Create and start the APScheduler background scheduler.

    Registers a single cron job:
        **Monday at 09:00** in the configured timezone.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _scheduler, _start_time

    if _scheduler is not None and _scheduler.running:
        print("[NarrateKPI] ⏰ Scheduler already running, skipping start")
        return

    _scheduler = BackgroundScheduler(
        timezone=SCHEDULER_TIMEZONE,
        job_defaults={
            "coalesce": True,           # Combine missed runs
            "max_instances": 1,         # Never run concurrently
            "misfire_grace_time": 3600,  # Up to 1 h late is OK
        },
    )

    _scheduler.add_job(
        scheduled_generate_all_reports,
        trigger=CronTrigger(day_of_week="mon", hour=9, minute=0),
        id="weekly_report_generation",
        name="Weekly Report Generation (Mon 09:00)",
        replace_existing=True,
    )

    _scheduler.start()
    _start_time = datetime.now(timezone.utc)

    print(
        f"[NarrateKPI] ⏰ Scheduler started: weekly run Monday 09:00 "
        f"(timezone={SCHEDULER_TIMEZONE})",
        flush=True,
    )

    next_run = _scheduler.get_job("weekly_report_generation").next_run_time
    if next_run:
        print(
            f"[NarrateKPI] 📅 Next scheduled run: "
            f"{next_run.strftime('%Y-%m-%d %H:%M %Z')}",
            flush=True,
        )


def stop_scheduler() -> None:
    """Shut down the APScheduler background scheduler gracefully."""
    global _scheduler

    if _scheduler is None:
        return

    _scheduler.shutdown(wait=False)
    _scheduler = None
    print("[NarrateKPI] ⏰ Scheduler stopped", flush=True)


def get_scheduler_status() -> Dict[str, Any]:
    """Return a snapshot of the scheduler state for the API status endpoint.

    Returns
    -------
    dict
        Keys: ``running``, ``next_run``, ``active_jobs``, ``timezone``,
        ``uptime_seconds``.
    """
    if _scheduler is None or not _scheduler.running:
        return {
            "running": False,
            "next_run": None,
            "active_jobs": [],
            "timezone": SCHEDULER_TIMEZONE,
            "uptime_seconds": None,
        }

    job = _scheduler.get_job("weekly_report_generation")
    next_run = job.next_run_time.isoformat() if (job and job.next_run_time) else None

    uptime = None
    if _start_time:
        uptime = round((datetime.now(timezone.utc) - _start_time).total_seconds(), 1)

    return {
        "running": True,
        "next_run": next_run,
        "active_jobs": [job.name] if job else [],
        "timezone": SCHEDULER_TIMEZONE,
        "uptime_seconds": uptime,
    }
