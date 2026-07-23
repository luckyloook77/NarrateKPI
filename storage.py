"""
NarrateKPI — Persistent Report Store (Module 3)

Provides a JSON file-backed storage layer for managing report lifecycle
states: DRAFT → IN_REVIEW → APPROVED → SENT.

Thread-safe via a file-based lock pattern (simple for single-process
FastAPI with uvicorn).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


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
#  Report Record
# ──────────────────────────────────────────────────────────────────────

class ReportRecord:
    """In-memory representation of a single report with its lifecycle metadata.

    Serialised to/from JSON for persistence.
    """

    def __init__(
        self,
        *,
        report_id: Optional[str] = None,
        account_id: str,
        client_name: str,
        period: str,
        status: ReportStatus = ReportStatus.DRAFT,
        raw_metrics: Optional[Dict[str, Any]] = None,
        anomalies_found: int = 0,
        markdown_content: str = "",
        target_email: Optional[str] = None,
        sent_at: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        self.report_id = report_id or uuid4().hex[:12]
        self.account_id = account_id
        self.client_name = client_name
        self.period = period
        self.status = status
        self.raw_metrics = raw_metrics or {}
        self.anomalies_found = anomalies_found
        self.markdown_content = markdown_content
        self.target_email = target_email
        self.sent_at = sent_at
        now = _now_iso()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "id": self.report_id,
            "account_id": self.account_id,
            "client_name": self.client_name,
            "period": self.period,
            "status": self.status.value,
            "raw_metrics": self.raw_metrics,
            "anomalies_found": self.anomalies_found,
            "markdown_content": self.markdown_content,
            "target_email": self.target_email,
            "sent_at": self.sent_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportRecord":
        """Deserialise from a dictionary (as stored in JSON)."""
        return cls(
            report_id=data.get("id"),
            account_id=data.get("account_id", ""),
            client_name=data.get("client_name", ""),
            period=data.get("period", ""),
            status=ReportStatus(data.get("status", "DRAFT")),
            raw_metrics=data.get("raw_metrics", {}),
            anomalies_found=data.get("anomalies_found", 0),
            markdown_content=data.get("markdown_content", ""),
            target_email=data.get("target_email"),
            sent_at=data.get("sent_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────
#  JSON File Store
# ──────────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent

_DEFAULT_STORE = _BASE_DIR / "reports_store.json"

# Use NARRATEKPI_STORE_PATH env var if set; otherwise default to
# reports_store.json in the project root (always writable).
_raw = os.environ.get("NARRATEKPI_STORE_PATH", None)
if _raw:
    _candidate = Path(_raw)
    if _candidate.is_absolute():
        # Absolute path — verify the parent dir is writable at startup.
        try:
            _candidate.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            print(
                f"[NarrateKPI] ⚠️  STORE_PATH '{_raw}' parent dir not writable. "
                f"Falling back to '{_DEFAULT_STORE}'.",
                flush=True,
            )
            _candidate = _DEFAULT_STORE
    else:
        _candidate = _BASE_DIR / _raw
    STORE_PATH = str(_candidate)
else:
    STORE_PATH = str(_DEFAULT_STORE)

_lock = threading.Lock()


def _read_store() -> List[Dict[str, Any]]:
    """Read all records from the JSON file."""
    path = Path(STORE_PATH)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_store(records: List[Dict[str, Any]]) -> None:
    """Overwrite the JSON file with the full record list."""
    path = Path(STORE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        print(
            f"[NarrateKPI] ❌ Cannot create store directory '{path.parent}': {exc}. "
            f"Falling back to '{_DEFAULT_STORE}' for this write.",
            flush=True,
        )
        # Write to the fallback path for this call only (don't mutate the global).
        path = _DEFAULT_STORE
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def load_all() -> List[ReportRecord]:
    """Load every report record from the store."""
    with _lock:
        return [ReportRecord.from_dict(d) for d in _read_store()]


def load_by_id(report_id: str) -> Optional[ReportRecord]:
    """Load a single report by its ID."""
    with _lock:
        for d in _read_store():
            if d.get("id") == report_id:
                return ReportRecord.from_dict(d)
    return None


def save(record: ReportRecord) -> ReportRecord:
    """Insert or update a report record.

    If a record with the same ``report_id`` already exists it is replaced;
    otherwise appended.
    """
    record.updated_at = _now_iso()
    with _lock:
        records = _read_store()
        new_dict = record.to_dict()
        for i, d in enumerate(records):
            if d.get("id") == record.report_id:
                records[i] = new_dict
                break
        else:
            records.append(new_dict)
        _write_store(records)
    return record


def delete(report_id: str) -> bool:
    """Remove a report by ID. Returns ``True`` if found and removed."""
    with _lock:
        records = _read_store()
        new_records = [d for d in records if d.get("id") != report_id]
        if len(new_records) == len(records):
            return False
        _write_store(new_records)
    return True


def clear_all() -> None:
    """Wipe the store (for testing / regenerating)."""
    with _lock:
        _write_store([])
