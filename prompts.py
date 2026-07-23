"""
NarrateKPI — LLM Prompts & Templates (Module 2)

Holds the battle-tested system prompt for the LLM, the user-prompt builder
that injects ``summary_raw_json``, and a template-based mock report generator
used when no API key is configured (dry-run mode).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


# ──────────────────────────────────────────────────────────────────────
#  System Prompt
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Senior Agency Account Manager & Performance Marketing Strategist.
You are responsible for communicating weekly performance reports to clients in a clear, actionable, and professional manner.

## CONSTRAINTS — YOU MUST FOLLOW THESE STRICTLY:

1. **DO NOT alter, invent, or round any numbers.** Use the exact figures provided in the JSON payload. Every metric value, percentage change, and anomaly count must appear verbatim as supplied.

2. **Explain technical jargon.** Every metric acronym (CPA, ROAS, CTR, CPC, CPM, etc.) must be followed by a brief plain-language explanation in parentheses the first time it appears in each section. Example: "CPA (cost per acquisition, i.e. the average cost to generate one customer lead)".

3. **Output structure is mandatory.** Your entire response MUST follow this Markdown format with exactly these three sections and headings:

---

## 📊 Executive Summary

Write 2-3 sentences summarising overall performance: how the account is trending, the total budget spent vs the previous period, and whether the overall direction is healthy, concerning, or mixed.

## 🔍 Key Insights & What Happened

For EVERY anomaly flagged in the JSON payload, write a short paragraph explaining:
- What happened (using exact numbers from the payload).
- WHY it likely happened (e.g. auction dynamics, creative fatigue, seasonality, landing page experience, audience saturation).
- What metric was affected and how it connects to the broader account health.

Group anomalies by severity: start with CRITICAL warnings, then WARNING items, then POSITIVE findings, then NEUTRAL observations. If no anomalies were found, state that the account is stable and note the most important directional moves in the raw metrics.

## 🚀 Action Plan for Next Week

Write exactly 3 actionable steps. Each step must:
- Be specific (name the channel, campaign type, or tactic).
- Address a specific issue or opportunity identified in the Key Insights section.
- State what the agency team will do and what the expected outcome should be.

---

4. **Tone**: Professional, confident, data-driven. Write as a partner who deeply understands the client's business. Use "we" and "your" appropriately.

5. **No markdown inside markdown**: Do not wrap the output in code fences. The response IS the report."""


# ──────────────────────────────────────────────────────────────────────
#  User-Prompt Builder
# ──────────────────────────────────────────────────────────────────────

def build_user_prompt(summary_raw_json: Dict[str, Any]) -> str:
    """Construct the user message by embedding the JSON payload.

    Parameters
    ----------
    summary_raw_json:
        The ``summary_raw_json`` dict from an ``AnomalyReport``, produced
        by ``math_engine.run_math_engine()``.

    Returns
    -------
    str
        A formatted user prompt ready to send to the LLM.
    """
    client_name = summary_raw_json.get("client_name", "Unknown Client")
    period_current = summary_raw_json.get("period_current", "N/A")
    period_previous = summary_raw_json.get("period_previous", "N/A")

    # Build a concise metrics summary table for the prompt.
    cur = summary_raw_json.get("current_metrics", {})
    prev = summary_raw_json.get("previous_metrics", {})
    anomalies = summary_raw_json.get("anomalies", [])

    lines = [
        f"Generate a weekly performance report for **{client_name}**.",
        f"Period: **{period_current}** vs **{period_previous}**.\n",
        "## Raw Metrics\n",
        f"| Metric | Current ({period_current}) | Previous ({period_previous}) |",
        "|--------|---------------------------|-----------------------------|",
    ]

    # Metrics table.
    metric_order = [
        ("Impressions", "impressions", int),
        ("Clicks", "clicks", int),
        ("Spend ($)", "spend", float),
        ("Conversions", "conversions", int),
        ("Revenue ($)", "revenue", float),
        ("CTR (%)", "ctr", float),
        ("CPC ($)", "cpc", float),
        ("CPA ($)", "cpa", float),
        ("ROAS ($)", "roas", float),
    ]
    for label, key, _ in metric_order:
        c = cur.get(key, "—")
        p = prev.get(key, "—")
        # Format numbers nicely.
        c_str = f"{c:,.4f}" if isinstance(c, float) else f"{c:,}"
        p_str = f"{p:,.4f}" if isinstance(p, float) else f"{p:,}"
        lines.append(f"| {label} | {c_str} | {p_str} |")

    lines.append("")

    # Anomalies section.
    if anomalies:
        lines.append("## Detected Anomalies\n")
        lines.append(f"Total anomalies found: {summary_raw_json.get('total_anomalies_found', 0)}\n")
        for i, anomaly in enumerate(anomalies, 1):
            lines.append(f"### Anomaly {i}: {anomaly.get('metric_name', 'N/A')} ({anomaly.get('severity', 'N/A')})")
            lines.append(f"- Current value: {anomaly.get('current_value', 'N/A')}")
            lines.append(f"- Previous value: {anomaly.get('previous_value', 'N/A')}")
            lines.append(f"- Percent change: {anomaly.get('percent_change', 'N/A'):+.2f}%")
            lines.append(f"- Direction: {anomaly.get('direction', 'N/A')}")
            lines.append(f"- Message: {anomaly.get('message', 'N/A')}")
            lines.append("")
    else:
        lines.append("## No Anomalies Detected\n")
        lines.append("All metrics are within normal thresholds. Minor fluctuations are expected.\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  Mock / Dry-Run Report Generator
# ──────────────────────────────────────────────────────────────────────

def generate_mock_report(summary_raw_json: Dict[str, Any]) -> str:
    """Generate a realistic, structured Markdown report **without an LLM**.

    Used as a fallback when no API key is configured. The output follows the
    same structure the LLM would produce (Executive Summary → Key Insights →
    Action Plan) so the pipeline can be tested end-to-end without burning
    API credits.
    """
    period_cur: str = summary_raw_json.get("period_current", "current")
    period_prev: str = summary_raw_json.get("period_previous", "previous")
    cur: Dict = summary_raw_json.get("current_metrics", {})
    prev: Dict = summary_raw_json.get("previous_metrics", {})
    anomalies: List[Dict] = summary_raw_json.get("anomalies", [])
    total_anomalies = summary_raw_json.get("total_anomalies_found", 0)

    spend_cur = cur.get("spend", 0)
    spend_prev = prev.get("spend", 0)
    revenue_cur = cur.get("revenue", 0)
    revenue_prev = prev.get("revenue", 0)
    conv_cur = cur.get("conversions", 0)
    conv_prev = prev.get("conversions", 0)

    # ── Executive Summary ──────────────────────────────────────────
    if total_anomalies == 0:
        exec_lines = [
            f"Your account remained stable during **{period_cur}** with no metrics "
            f"breaching our alert thresholds. Total spend was **${spend_cur:,.2f}** "
            f"(vs ${spend_prev:,.2f} the prior week), generating "
            f"**{conv_cur:,} conversions** and **${revenue_cur:,.2f} in revenue**. "
            "Performance is within expected ranges, and no immediate action is required.",
        ]
    else:
        critical_count = sum(1 for a in anomalies if a.get("severity") == "CRITICAL")
        positive_count = sum(1 for a in anomalies if a.get("severity") == "POSITIVE")
        warning_count = sum(1 for a in anomalies if a.get("severity") == "WARNING")

        parts = []
        if critical_count:
            parts.append(f"{critical_count} critical issue{'s' if critical_count > 1 else ''}")
        if warning_count:
            parts.append(f"{warning_count} warning{'s' if warning_count > 1 else ''}")
        if positive_count:
            parts.append(f"{positive_count} positive signal{'s' if positive_count > 1 else ''}")

        status = "mixed performance" if critical_count else "strong performance"
        exec_lines = [
            f"Your account showed **{status}** during **{period_cur}** with "
            f"{' and '.join(parts)} detected. "
            f"Total spend was **${spend_cur:,.2f}** "
            f"(vs ${spend_prev:,.2f} the prior week), generating "
            f"**{conv_cur:,} conversions** and **${revenue_cur:,.2f} in revenue**. "
        ]

    # ── Key Insights ───────────────────────────────────────────────
    insight_sections = []

    # Sort anomalies by severity priority.
    severity_order = {"CRITICAL": 0, "WARNING": 1, "POSITIVE": 2, "NEUTRAL": 3}
    sorted_anomalies = sorted(
        anomalies, key=lambda a: severity_order.get(a.get("severity", "NEUTRAL"), 99)
    )

    for anomaly in sorted_anomalies:
        metric = anomaly.get("metric_name", "")
        severity = anomaly.get("severity", "NEUTRAL")
        pct = anomaly.get("percent_change", 0.0)
        # Guard against infinity (possible when previous_value was 0).
        pct_display = f"{pct:+.2f}%" if math.isfinite(pct) else "N/A (previous period was zero)"
        cur_val = anomaly.get("current_value", 0)
        prev_val = anomaly.get("previous_value", 0)
        msg = anomaly.get("message", "")

        icon = {"CRITICAL": "🚨", "WARNING": "⚠️", "POSITIVE": "✅", "NEUTRAL": "ℹ️"}.get(severity, "ℹ️")
        severity_label = {"CRITICAL": "Critical", "WARNING": "Warning", "POSITIVE": "Positive", "NEUTRAL": "Neutral"}.get(severity, "Info")

        paragraph = f"**{icon} {severity_label}: {metric}** — {msg}"

        # Add a "why" narrative based on the metric and direction.
        if metric == "CPA":
            if pct > 0:
                paragraph += (
                    " This increase in cost per acquisition may be driven by higher "
                    "auction competition, audience saturation, or less efficient ad placements. "
                    "We recommend reviewing the ad-set-level cost data to identify the source."
                )
            else:
                paragraph += (
                    " This decline in cost per acquisition is a strong efficiency signal, "
                    "likely driven by improved audience targeting, creative resonance, or "
                    "favourable auction dynamics. We should identify and scale the best performers."
                )
        elif metric == "ROAS":
            if pct > 0:
                paragraph += (
                    " This return-on-ad-spend improvement means every dollar invested is "
                    "generating more revenue. Strong creative performance, better audience matching, "
                    "or favourable seasonality may be contributing."
                )
            else:
                paragraph += (
                    " A declining ROAS (return on ad spend) suggests the revenue generated per dollar "
                    "invested is shrinking. This could indicate creative fatigue, increased competition, "
                    "or a shift in conversion behaviour lower in the funnel."
                )
        elif metric == "CTR":
            if pct < 0:
                paragraph += (
                    " A lower click-through rate may signal ad fatigue or reduced relevance. "
                    "Consider refreshing creative assets, testing new ad copy, or re-evaluating "
                    "audience targeting parameters."
                )
        elif metric == "Spend+Conversions":
            paragraph += (
                " Budget is increasing without proportional conversion growth, indicating "
                "inefficient spend. We recommend pausing or reducing budget on under-performing "
                "ad sets and reallocating to higher-efficiency campaigns."
            )

        insight_sections.append(f"### {icon} {metric}\n\n{paragraph}\n")

    if not anomalies:
        # No anomalies — highlight the most important metric movements.
        notable = []
        for label, key in [("Impressions", "impressions"), ("Clicks", "clicks"),
                           ("Spend", "spend"), ("Conversions", "conversions"),
                           ("Revenue", "revenue"), ("CTR", "ctr"),
                           ("CPC", "cpc"), ("CPA", "cpa"), ("ROAS", "roas")]:
            c = cur.get(key, 0)
            p = prev.get(key, 0)
            if isinstance(c, (int, float)) and isinstance(p, (int, float)) and p:
                pct = (c - p) / p * 100
                if abs(pct) > 5:
                    direction = "increased" if pct > 0 else "decreased"
                    notable.append(f"{label} {direction} by {abs(pct):.1f}%")
        if notable:
            insight_sections.append(
                "### 📊 Notable Metric Movements\n\n"
                "While no anomalies were triggered, the following metrics moved materially:\n\n"
                + "\n".join(f"- {n}" for n in notable) + "\n"
            )
        else:
            insight_sections.append(
                "### 📊 Stable Account\n\n"
                "All metrics remained within expected ranges. "
                "No material changes to report.\n"
            )

    # ── Action Plan ────────────────────────────────────────────────
    actions = _build_mock_actions(anomalies, cur, prev)

    # ── Assemble ───────────────────────────────────────────────────
    sections = [
        "## 📊 Executive Summary",
        "",
        *exec_lines,
        "",
        "---",
        "",
        "## 🔍 Key Insights & What Happened",
        "",
        *insight_sections,
        "---",
        "",
        "## 🚀 Action Plan for Next Week",
        "",
        *actions,
        "",
    ]

    # Add a footer note about this being a mock report.
    sections.extend([
        "",
        "---",
        "",
        "> *This report was generated in dry-run mode. No LLM API was called.*",
        "",
    ])

    return "\n".join(sections)


def _build_mock_actions(
    anomalies: List[Dict[str, Any]],
    cur: Dict[str, Any],
    prev: Dict[str, Any],
) -> List[str]:
    """Generate 3 mock action items based on detected anomalies."""
    actions: List[str] = []

    severity_order = {"CRITICAL": 0, "WARNING": 1, "POSITIVE": 2}
    sorted_anomalies = sorted(
        anomalies, key=lambda a: severity_order.get(a.get("severity", ""), 99)
    )

    # Collect distinct metric names that triggered.
    triggered_metrics = {a.get("metric_name") for a in sorted_anomalies}

    # ---- Action 1 ----
    if triggered_metrics & {"CPA", "Spend+Conversions"}:
        actions.append(
            "**1. Audit & Optimise Under-Performing Ad Sets**\n\n"
            "The CPA spike and inefficient spend-growth pattern indicate that some ad sets are "
            "driving up costs without proportional returns. Our team will conduct a campaign-level "
            "audit to identify the specific ad sets with the highest CPA and lowest conversion rates. "
            "We will pause or reduce budget on bottom-quartile performers and reallocate spend to "
            "ad sets with healthy efficiency metrics. Expected outcome: CPA reduction of "
            "15–25% and improved overall ROAS within the next reporting period."
        )
    elif triggered_metrics & {"ROAS"} and any(
        a.get("severity") == "CRITICAL" for a in anomalies if a.get("metric_name") == "ROAS"
    ):
        actions.append(
            "**1. Investigate Revenue Efficiency Decline**\n\n"
            "ROAS has dropped below our efficiency threshold, signalling that each dollar invested "
            "is generating less revenue. Our team will analyse funnel performance to identify "
            "where conversions are dropping off — whether at the landing page, checkout, or "
            "lead-qualification stage. We will test updated creative assets and audience "
            "refinements to restore revenue efficiency. Expected outcome: ROAS recovery toward "
            "previous-period levels within two weeks."
        )
    elif triggered_metrics & {"CTR"}:
        actions.append(
            "**1. Refresh Creative Assets**\n\n"
            "The declining click-through rate suggests ad fatigue may be setting in. Our creative "
            "team will produce 3-5 new ad variants (new copy, visuals, and calls-to-action) "
            "to test against the current creatives. We will A/B test these over the next 7 days "
            "and scale the winning variants. Expected outcome: CTR improvement of 10–20% and "
            "reduced cost per click."
        )
    elif triggered_metrics & {"CPA", "ROAS"} and all(
        a.get("severity") == "POSITIVE" for a in anomalies
    ):
        actions.append(
            "**1. Scale Winning Campaigns**\n\n"
            "With both CPA declining and ROAS surging, we have clear signal on which campaigns "
            "are delivering exceptional efficiency. Our team will increase budget allocation to "
            "the top-performing campaigns by 20–30%, while closely monitoring frequency and "
            "cost metrics to avoid diminishing returns. Expected outcome: increased volume at "
            "maintained efficiency levels."
        )
    else:
        actions.append(
            "**1. Maintain Current Strategy with Light Optimisation**\n\n"
            "With no critical anomalies detected, the current campaign strategy is performing "
            "within expected parameters. Our team will continue monitoring key metrics and "
            "perform minor bid adjustments to capture incremental efficiency gains. "
            "Expected outcome: sustained performance with gradual improvement."
        )

    # ---- Action 2 ----
    if triggered_metrics & {"ROAS", "CPA"} or "Spend+Conversions" in triggered_metrics:
        actions.append(
            "**2. Implement Bid & Budget Adjustments**\n\n"
            "Based on the efficiency signals identified, our team will adjust bid strategies "
            "to prioritise high-converting audience segments. We will implement dayparting "
            "adjustments to concentrate spend during peak conversion hours and apply audience "
            "layers to exclude low-intent traffic. Expected outcome: improved cost efficiency "
            "and higher conversion rates."
        )
    else:
        actions.append(
            "**2. Explore New Audience & Creative Opportunities**\n\n"
            "With stable account performance, we have capacity to test new growth levers. "
            "Our team will launch a small-scale budget test (10% of total spend) targeting "
            "a new lookalike audience segment and test 2 new creative concepts. "
            "Expected outcome: identify new scalable acquisition channels without risking "
            "core campaign performance."
        )

    # ---- Action 3 ----
    if any(a.get("severity") == "CRITICAL" for a in anomalies):
        actions.append(
            "**3. Set Up Automated Budget Caps & Alerts**\n\n"
            "To prevent similar cost surges in the future, our team will configure automated "
            "budget caps and cost-per-result alert rules at the campaign level. This ensures "
            "that if CPA or spend efficiency breaches safe thresholds mid-week, the system "
            "will automatically reduce spend until our team can investigate. "
            "Expected outcome: proactive cost control and reduced risk of budget overrun."
        )
    elif any(a.get("severity") == "POSITIVE" for a in anomalies):
        actions.append(
            "**3. Document & Replicate Winning Patterns**\n\n"
            "Our team will document the creative, targeting, and bidding elements that drove "
            "the positive performance this week. These insights will be used to inform campaign "
            "structures for upcoming launches and shared with the broader account team. "
            "Expected outcome: institutionalised best practices for sustained efficiency gains."
        )
    else:
        actions.append(
            "**3. Prepare for Next-Week Campaign Calendar**\n\n"
            "Our team will align with you on upcoming promotions, product launches, or "
            "seasonal events so we can adjust campaign calendars, budgets, and creative "
            "accordingly. Proactive planning ensures we capture demand spikes while "
            "maintaining cost discipline. Expected outcome: campaign readiness and "
            "optimised resource allocation."
        )

    return actions
