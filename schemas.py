"""
NarrateKPI — Core Data Schemas (Module 1)

Defines the Pydantic v2 models, enums, and type aliases used by the Math Engine
and downstream LLM-context generation.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Dict, List, Optional

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias  # pragma: no cover

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────────────────

class AnomalySeverity(str, Enum):
    """Severity classification for an anomaly."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"


class MetricDirection(str, Enum):
    """Direction of metric change compared to the previous period."""

    INCREASED = "INCREASED"
    DECREASED = "DECREASED"
    UNCHANGED = "UNCHANGED"


# ──────────────────────────────────────────────────────────────────────
#  Type aliases
# ──────────────────────────────────────────────────────────────────────

AccountID: TypeAlias = str
"""UUID or human-readable account identifier."""

ISOWeekPeriod: TypeAlias = str
"""ISO week string such as ``"2026-W29"``."""

JSONDict: TypeAlias = Dict[str, object]
"""Generic JSON-serialisable dictionary."""


# ──────────────────────────────────────────────────────────────────────
#  Metric models
# ──────────────────────────────────────────────────────────────────────

class MetricSet(BaseModel):
    """Raw or derived advertising metrics for a single time period.

    Fields that can be auto-calculated (CTR, CPC, CPA, ROAS) are optional
    so the engine can fill them in when they are missing or zero.
    """

    impressions: int = Field(..., ge=0, description="Total ad impressions")
    clicks: int = Field(..., ge=0, description="Total ad clicks")
    spend: float = Field(..., ge=0.0, description="Total media spend")
    conversions: int = Field(..., ge=0, description="Total conversions")
    revenue: float = Field(..., ge=0.0, description="Attributed revenue")
    ctr: Optional[float] = Field(default=None, ge=0.0, description="Click-through rate (%)")
    cpc: Optional[float] = Field(default=None, ge=0.0, description="Cost per click")
    cpa: Optional[float] = Field(default=None, ge=0.0, description="Cost per acquisition")
    roas: Optional[float] = Field(default=None, ge=0.0, description="Return on ad spend (×)")


class Anomaly(BaseModel):
    """A single detected anomaly for one metric."""

    metric_name: str = Field(..., description="Name of the affected metric")
    current_value: float = Field(..., description="Value in the current period")
    previous_value: float = Field(..., description="Value in the comparison period")
    percent_change: float = Field(..., description="Percentage change (signed)")
    absolute_change: float = Field(..., description="Absolute difference (signed)")
    direction: MetricDirection = Field(..., description="Direction of change")
    severity: AnomalySeverity = Field(..., description="Severity level")
    message: str = Field(..., description="Human-readable anomaly explanation")


# ──────────────────────────────────────────────────────────────────────
#  Domain models
# ──────────────────────────────────────────────────────────────────────

class WeeklyComparisonInput(BaseModel):
    """Input payload encapsulating a week-over-week comparison."""

    account_id: AccountID = Field(..., description="Unique account identifier")
    client_name: str = Field(..., description="Human-readable client name")
    period_current: ISOWeekPeriod = Field(..., description="Current ISO week, e.g. 2026-W29")
    period_previous: ISOWeekPeriod = Field(..., description="Previous ISO week, e.g. 2026-W28")
    current_metrics: MetricSet = Field(..., description="Metrics for the current period")
    previous_metrics: MetricSet = Field(..., description="Metrics for the comparison period")


class AnomalyReport(BaseModel):
    """Output of the Math Engine — a report containing every detected anomaly.

    ``summary_raw_json`` is designed to be passed directly into an LLM prompt
    as the context payload so the LLM can generate natural-language commentary.
    """

    account_id: AccountID = Field(..., description="Unique account identifier")
    client_name: str = Field(..., description="Human-readable client name")
    total_anomalies_found: int = Field(..., ge=0, description="Count of anomalies detected")
    anomalies: List[Anomaly] = Field(default_factory=list, description="All detected anomalies")
    summary_raw_json: JSONDict = Field(
        default_factory=dict,
        description="LLM-ready context payload (dict representation of this report)",
    )
