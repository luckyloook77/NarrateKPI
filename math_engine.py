"""
NarrateKPI — Mathematical Anomaly Detection Engine (Module 1)

Pure deterministic calculations for:
  • Safe percentage-change computation (handles zero-division).
  • Auto-calculation of missing / zero-valued derived metrics.
  • Configurable threshold-rule anomaly detection.
  • Generation of LLM-ready ``AnomalyReport`` objects.

No external API calls, no LLM calls — every number is derived locally.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias  # pragma: no cover

from schemas import (
    Anomaly,
    AnomalyReport,
    AnomalySeverity,
    MetricDirection,
    MetricSet,
    WeeklyComparisonInput,
)


# ──────────────────────────────────────────────────────────────────────
#  Type aliases for trigger callables
# ──────────────────────────────────────────────────────────────────────

SimpleTrigger: TypeAlias = Callable[[float, float, float], bool]
"""(pct_change, current_scalar, previous_scalar) -> bool"""

CompositeTrigger: TypeAlias = Callable[[float, MetricSet, MetricSet], bool]
"""(pct_change, current_metrics, previous_metrics) -> bool"""



# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_pct_change(current: float, previous: float) -> Tuple[float, float]:
    """Compute percentage change and absolute change safely.

    Returns
    -------
    (percent_change, absolute_change)
        ``percent_change`` is ``inf`` or ``0.0`` when ``previous`` is zero.

    Formula
    -------
    .. math::

        \\Delta\\% = \\frac{\\text{current} - \\text{previous}}{\\text{previous}} \\times 100
    """
    abs_change = current - previous
    if previous == 0.0:
        # If previous is zero and current is non-zero, treat as a 100% move.
        pct = math.copysign(float("inf"), abs_change) if abs_change != 0.0 else 0.0
    else:
        pct = (abs_change / previous) * 100.0
    return (pct, abs_change)


def _round_significant(value: float, decimals: int = 2) -> float:
    """Round to *decimals* places, handling infinities gracefully."""
    if math.isfinite(value):
        return round(value, decimals)
    return value


# ──────────────────────────────────────────────────────────────────────
#  Derived-metric auto-calculation
# ──────────────────────────────────────────────────────────────────────

def auto_calculate(metrics: MetricSet) -> MetricSet:
    """Fill missing / zero-valued derived metrics (CTR, CPC, CPA, ROAS).

    The function updates **only** fields that are ``None`` or ``0``,
    leaving explicitly-provided values untouched.
    """
    overrides: dict = {}

    if not metrics.ctr:
        # CTR (%) = (clicks / impressions) × 100
        overrides["ctr"] = (
            round((metrics.clicks / metrics.impressions) * 100.0, 4)
            if metrics.impressions
            else 0.0
        )

    if not metrics.cpc:
        # CPC = spend / clicks
        overrides["cpc"] = (
            round(metrics.spend / metrics.clicks, 4)
            if metrics.clicks
            else 0.0
        )

    if not metrics.cpa:
        # CPA = spend / conversions
        overrides["cpa"] = (
            round(metrics.spend / metrics.conversions, 4)
            if metrics.conversions
            else 0.0
        )

    if not metrics.roas:
        # ROAS = revenue / spend
        overrides["roas"] = (
            round(metrics.revenue / metrics.spend, 4)
            if metrics.spend
            else 0.0
        )

    return metrics.model_copy(update=overrides) if overrides else metrics


# ──────────────────────────────────────────────────────────────────────
#  Threshold rules
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdRule:
    """A single anomaly-detection rule.

    Parameters
    ----------
    metric_name:
        Human-readable name (e.g. ``"CPA"``).
    get_current:
        Extracts the current-period value from a ``MetricSet``.
    get_previous:
        Extracts the comparison-period value from a ``MetricSet``.
    is_triggered:
        Callable.  For simple rules (``composite=False``), receives
        ``(pct_change, current_scalar, previous_scalar) -> bool``.
        For composite rules (``composite=True``), receives
        ``(pct_change, current_metrics, previous_metrics) -> bool``.
    severity:
        Severity to assign when triggered.
    composite:
        When ``True``, ``is_triggered`` receives the full ``MetricSet``
        objects instead of scalar values (for multi-metric rules).
    message_template:
        ``str.format`` template receiving ``metric_name``, ``current_val``,
        ``previous_val``, ``pct_change``, ``abs_change``, ``direction``,
        and ``severity``.
    """

    metric_name: str
    get_current: Callable[[MetricSet], float]
    get_previous: Callable[[MetricSet], float]
    is_triggered: Union[SimpleTrigger, CompositeTrigger]
    severity: AnomalySeverity
    composite: bool = False
    message_template: str = field(
        default=(
            "{metric_name} {direction_text} by {pct_change:.1f}% "
            "({previous_val:.2f} → {current_val:.2f}) — {severity.value}"
        )
    )


# ──────────────────────────────────────────────────────────────────────
#  Default rule set
# ──────────────────────────────────────────────────────────────────────

def _cpa_spike_trigger(pct: float, _cur: float, _prev: float) -> bool:
    return pct > 20.0


def _cpa_drop_trigger(pct: float, _cur: float, _prev: float) -> bool:
    return pct < -15.0


def _roas_drop_trigger(pct: float, _cur: float, _prev: float) -> bool:
    return pct < -15.0


def _roas_spike_trigger(pct: float, _cur: float, _prev: float) -> bool:
    return pct > 15.0


def _ctr_drop_trigger(pct: float, _cur: float, _prev: float) -> bool:
    return pct < -15.0


def _spend_spike_plus_conversions_drop(
    pct: float, cur: MetricSet, prev: MetricSet
) -> bool:
    """Composite rule: spend >+25% AND conversions dropped."""
    if pct <= 25.0:
        return False
    conv_pct, _ = _safe_pct_change(cur.conversions, prev.conversions)
    return conv_pct < 0.0


DEFAULT_RULES: List[ThresholdRule] = [
    ThresholdRule(
        metric_name="CPA",
        get_current=lambda m: m.cpa or 0.0,
        get_previous=lambda m: m.cpa or 0.0,
        is_triggered=_cpa_spike_trigger,
        severity=AnomalySeverity.CRITICAL,
        message_template=(
            "🚨 CPA spiked {pct_change:.1f}% ({previous_val:.2f} → {current_val:.2f}) "
            "— costs are rising sharply; review ad sets and keywords."
        ),
    ),
    ThresholdRule(
        metric_name="CPA",
        get_current=lambda m: m.cpa or 0.0,
        get_previous=lambda m: m.cpa or 0.0,
        is_triggered=_cpa_drop_trigger,
        severity=AnomalySeverity.POSITIVE,
        message_template=(
            "✅ CPA dropped {pct_change:.1f}% ({previous_val:.2f} → {current_val:.2f}) "
            "— costs are declining; identify winning placements."
        ),
    ),
    ThresholdRule(
        metric_name="ROAS",
        get_current=lambda m: m.roas or 0.0,
        get_previous=lambda m: m.roas or 0.0,
        is_triggered=_roas_drop_trigger,
        severity=AnomalySeverity.CRITICAL,
        message_template=(
            "🚨 ROAS dropped {pct_change:.1f}% ({previous_val:.2f} → {current_val:.2f}) "
            "— revenue efficiency is falling; investigate campaign performance."
        ),
    ),
    ThresholdRule(
        metric_name="ROAS",
        get_current=lambda m: m.roas or 0.0,
        get_previous=lambda m: m.roas or 0.0,
        is_triggered=_roas_spike_trigger,
        severity=AnomalySeverity.POSITIVE,
        message_template=(
            "✅ ROAS surged {pct_change:.1f}% ({previous_val:.2f} → {current_val:.2f}) "
            "— excellent revenue efficiency; scale winning campaigns."
        ),
    ),
    ThresholdRule(
        metric_name="CTR",
        get_current=lambda m: m.ctr or 0.0,
        get_previous=lambda m: m.ctr or 0.0,
        is_triggered=_ctr_drop_trigger,
        severity=AnomalySeverity.WARNING,
        message_template=(
            "⚠️ CTR dropped {pct_change:.1f}% ({previous_val:.2f} → {current_val:.2f}) "
            "— ad relevance may be declining; review creative assets."
        ),
    ),
    ThresholdRule(
        metric_name="Spend+Conversions",
        get_current=lambda m: float(m.spend),
        get_previous=lambda m: float(m.spend),
        is_triggered=_spend_spike_plus_conversions_drop,
        severity=AnomalySeverity.CRITICAL,
        composite=True,
        message_template=(
            "🚨 Spend surged {pct_change:.1f}% while Conversions dropped — "
            "budget is increasing without results; pause under-performing ad sets."
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────────
#  Engine
# ──────────────────────────────────────────────────────────────────────

def build_direction(pct_change: float) -> MetricDirection:
    """Classify the direction of change."""
    if pct_change > 0.0:
        return MetricDirection.INCREASED
    if pct_change < 0.0:
        return MetricDirection.DECREASED
    return MetricDirection.UNCHANGED


def _check_rule(
    rule: ThresholdRule,
    cur: MetricSet,
    prev: MetricSet,
) -> Optional[Anomaly]:
    """Evaluate a single rule against current/previous metrics.

    Returns ``None`` when the rule is not triggered.
    """
    current_val = rule.get_current(cur)
    previous_val = rule.get_previous(prev)

    pct_change, abs_change = _safe_pct_change(current_val, previous_val)
    direction = build_direction(pct_change)

    triggered: bool
    if rule.composite:
        triggered = rule.is_triggered(pct_change, cur, prev)  # type: ignore[arg-type]
    else:
        triggered = rule.is_triggered(pct_change, current_val, previous_val)  # type: ignore[arg-type]

    if not triggered:
        return None

    return Anomaly(
        metric_name=rule.metric_name,
        current_value=_round_significant(current_val),
        previous_value=_round_significant(previous_val),
        percent_change=_round_significant(pct_change),
        absolute_change=_round_significant(abs_change),
        direction=direction,
        severity=rule.severity,
        message=rule.message_template.format(
            metric_name=rule.metric_name,
            current_val=current_val,
            previous_val=previous_val,
            pct_change=pct_change,
            abs_change=abs_change,
            direction=direction,
            direction_text=direction.value.lower(),
            severity=rule.severity,
        ),
    )


def run_math_engine(
    comparison: WeeklyComparisonInput,
    rules: Optional[List[ThresholdRule]] = None,
) -> AnomalyReport:
    """Run the full anomaly-detection pipeline.

    Steps
    -----
    1. Auto-calculate any missing derived metrics for both periods.
    2. Evaluate every threshold rule.
    3. Package results into an ``AnomalyReport`` with an LLM-ready summary.

    Parameters
    ----------
    comparison:
        The week-over-week input payload.
    rules:
        Custom rule set; falls back to ``DEFAULT_RULES`` when ``None``.

    Returns
    -------
    AnomalyReport
        Fully populated report.
    """
    cur = auto_calculate(comparison.current_metrics)
    prev = auto_calculate(comparison.previous_metrics)

    applied_rules = rules if rules is not None else DEFAULT_RULES

    anomalies: List[Anomaly] = []
    for rule in applied_rules:
        anomaly = _check_rule(rule, cur, prev)
        if anomaly is not None:
            anomalies.append(anomaly)

    report = AnomalyReport(
        account_id=comparison.account_id,
        client_name=comparison.client_name,
        total_anomalies_found=len(anomalies),
        anomalies=anomalies,
    )

    # Build the LLM-ready summary payload (plain dict, no Pydantic models).
    report.summary_raw_json = {
        "account_id": report.account_id,
        "client_name": report.client_name,
        "period_current": comparison.period_current,
        "period_previous": comparison.period_previous,
        "total_anomalies_found": report.total_anomalies_found,
        "anomalies": [a.model_dump() for a in report.anomalies],
        "current_metrics": cur.model_dump(),
        "previous_metrics": prev.model_dump(),
    }

    return report
