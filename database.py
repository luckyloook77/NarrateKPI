"""
NarrateKPI — SQLAlchemy Database Layer (Module 6)

Replaces the JSON file store (``storage.py``) with a proper relational DB.
Supports PostgreSQL (via ``DATABASE_URL`` env var) and falls back to SQLite
for local development.

Models
------
- **Client** — Registry of agency clients (populated from report data).
- **Report** — Full report record with metrics, markdown, and lifecycle status.

Migration
---------
On first startup, ``migrate_from_json()`` seeds the DB from the existing
``reports_store.json`` file, then backs it up as ``.json.bak``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import JSON, Column, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


# ──────────────────────────────────────────────────────────────────────
#  Report Status Lifecycle
# ──────────────────────────────────────────────────────────────────────


class ReportStatus(str, Enum):
    """Lifecycle state of a client report."""

    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    SENT = "SENT"


# ──────────────────────────────────────────────────────────────────────
#  Database URL resolution
# ──────────────────────────────────────────────────────────────────────


def _resolve_db_url() -> str:
    """Resolve ``DATABASE_URL`` with PostgreSQL→SQLAlchemy compatibility.

    Render's free Postgres exposes ``postgres://`` which SQLAlchemy < 2.0
    rejects.  This helper converts it to ``postgresql://`` automatically.

    Falls back to local SQLite when ``DATABASE_URL`` is not set.
    """
    raw = os.environ.get("DATABASE_URL")
    if raw:
        # postgres:// → postgresql:// (SQLAlchemy 1.x compat; harmless on 2.x)
        if raw.startswith("postgres://"):
            raw = raw.replace("postgres://", "postgresql://", 1)
        return raw
    # Local fallback — stores in project root, gitignored.
    return "sqlite:///./narratekpi.db"


# ──────────────────────────────────────────────────────────────────────
#  Engine & Session factory
# ──────────────────────────────────────────────────────────────────────

DATABASE_URL = _resolve_db_url()

_IS_SQLITE = "sqlite" in DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use (Render free tier restarts)
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ──────────────────────────────────────────────────────────────────────
#  Declarative base
# ──────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────────────────


class Client(Base):
    """Agency client registry, populated from generated reports."""

    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(128), unique=True, nullable=False, index=True)
    client_name = Column(String(255), nullable=False)
    target_email = Column(String(255), nullable=True)

    reports = relationship("Report", back_populates="client")

    def __repr__(self) -> str:
        return f"<Client {self.account_id}: {self.client_name}>"


class Report(Base):
    """A single weekly performance report with full lifecycle metadata."""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), unique=True, nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    account_id = Column(String(128), nullable=False)
    client_name = Column(String(255), nullable=False)  # denormalised for fast listing
    period = Column(String(16), nullable=False)
    status = Column(String(16), nullable=False, default=ReportStatus.DRAFT.value)
    raw_metrics = Column(JSON, nullable=False, default=dict)
    anomalies_found = Column(Integer, nullable=False, default=0)
    markdown_content = Column(Text, nullable=False, default="")
    target_email = Column(String(255), nullable=True)
    sent_at = Column(String(32), nullable=True)
    created_at = Column(String(32), nullable=False)
    updated_at = Column(String(32), nullable=False)

    client = relationship("Client", back_populates="reports")

    def __repr__(self) -> str:
        return f"<Report {self.report_id}: {self.client_name} ({self.status})>"


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp without trailing timezone offset."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────
#  FastAPI dependency
# ──────────────────────────────────────────────────────────────────────


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a DB session, close on teardown.

    Usage:
        .. code-block:: python

            from fastapi import Depends
            from sqlalchemy.orm import Session

            @app.get("/api/reports")
            def list_reports(db: Session = Depends(get_db)):
                ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────
#  Migration from JSON store
# ──────────────────────────────────────────────────────────────────────


def migrate_from_json(db: Session) -> int:
    """Seed the database from ``reports_store.json`` if it exists.

    Creates ``Client`` records for any new account IDs found and inserts
    corresponding ``Report`` rows.  The JSON file is renamed to
    ``.json.bak`` after a successful migration.

    Returns the number of reports migrated (0 if no JSON file was found).
    """
    base_dir = Path(__file__).resolve().parent
    store_path = base_dir / "reports_store.json"

    if not store_path.exists():
        return 0

    try:
        with open(store_path, "r", encoding="utf-8") as f:
            records: List[Dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[NarrateKPI] ⚠️  Could not read JSON store ({exc}) — skipping migration.")
        return 0

    if not isinstance(records, list) or not records:
        return 0

    count = 0
    for data in records:
        account_id = data.get("account_id", "")
        client_name = data.get("client_name", "")

        # ── Find or create client ──────────────────────────────────
        client = db.query(Client).filter(Client.account_id == account_id).first()
        if client is None:
            client = Client(
                account_id=account_id,
                client_name=client_name,
                target_email=data.get("target_email"),
            )
            db.add(client)
            db.flush()  # get client.id

        # ── Create report (skip if report_id already exists) ────────
        report_id = data.get("id", "")
        if not report_id:
            continue
        existing = db.query(Report).filter(Report.report_id == report_id).first()
        if existing:
            continue

        report = Report(
            report_id=report_id,
            client_id=client.id,
            account_id=account_id,
            client_name=client_name,
            period=data.get("period", ""),
            status=data.get("status", ReportStatus.DRAFT.value),
            raw_metrics=data.get("raw_metrics", {}),
            anomalies_found=data.get("anomalies_found", 0),
            markdown_content=data.get("markdown_content", ""),
            target_email=data.get("target_email"),
            sent_at=data.get("sent_at"),
            created_at=data.get("created_at") or _now_iso(),
            updated_at=data.get("updated_at") or _now_iso(),
        )
        db.add(report)
        count += 1

    db.commit()

    if count:
        # Backup and remove the JSON file so we don't re-import on restart.
        backup_path = store_path.with_suffix(".json.bak")
        store_path.rename(backup_path)
        print(f"[NarrateKPI] ✅ Migrated {count} report(s) from JSON store → DB")
        print(f"[NarrateKPI] 📦 JSON store backed up as {backup_path.name}")
    else:
        print(f"[NarrateKPI] ℹ️  JSON store found but no new records to migrate")

    return count


def init_db() -> Session:
    """Create all tables and run the JSON→DB migration.

    Call once at server startup (e.g. in a ``@app.on_event("startup")``
    handler).  Returns a session that the caller may close.
    """
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        migrated = migrate_from_json(db)
        if migrated == 0:
            print("[NarrateKPI] 🗄️  Database ready (no JSON migration performed)")
        return db
    except Exception:
        db.close()
        raise
