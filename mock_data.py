"""
NarrateKPI — Mock Data Generator (Module 1)

Provides three realistic, distinct test cases for the Math Engine:
  • Case A — E-Commerce Client (CPA spike, bad ad set performance)
  • Case B — B2B SaaS Client (strong ROAS & conversions up)
  • Case C — Stable Account (minor fluctuations, no major anomalies)
"""

from schemas import MetricSet, WeeklyComparisonInput


# ──────────────────────────────────────────────────────────────────────
#  Case A — E-Commerce Client
# ──────────────────────────────────────────────────────────────────────
# Scenario: A Shopify store running Meta advantage+ shopping campaigns.
# The previous week performed well, but the current week saw a CPA spike
# due to a poorly-tested ad set that drove expensive conversions.

case_a = WeeklyComparisonInput(
    account_id="acc_shopify_042",
    client_name="Urban Threads (E-Commerce)",
    period_current="2026-W29",
    period_previous="2026-W28",
    current_metrics=MetricSet(
        impressions=185_000,
        clicks=4_200,
        spend=8_750.00,
        conversions=85,
        revenue=21_000.00,
        # Derived metrics left as None so the engine auto-calculates.
    ),
    previous_metrics=MetricSet(
        impressions=210_000,
        clicks=5_100,
        spend=6_200.00,
        conversions=124,
        revenue=24_800.00,
    ),
)

# Expected: CPA will spike from ~50.00 to ~102.94 (+105.9%), triggering
# a CRITICAL anomaly. Spend is up while conversions dropped — the
# composite Spend+Conversions rule should also fire.


# ──────────────────────────────────────────────────────────────────────
#  Case B — B2B SaaS Client
# ──────────────────────────────────────────────────────────────────────
# Scenario: A LinkedIn / Meta lead-gen client with strong creative refresh.
# ROAS surged, CPA dropped sharply — a winning week.

case_b = WeeklyComparisonInput(
    account_id="acc_saas_119",
    client_name="DataFlow Analytics (B2B SaaS)",
    period_current="2026-W29",
    period_previous="2026-W28",
    current_metrics=MetricSet(
        impressions=95_000,
        clicks=3_800,
        spend=12_500.00,
        conversions=210,
        revenue=187_500.00,
    ),
    previous_metrics=MetricSet(
        impressions=102_000,
        clicks=3_400,
        spend=14_200.00,
        conversions=165,
        revenue=127_800.00,
    ),
)

# Expected: CPA dropped from ~86.06 to ~59.52 (-30.8%) → POSITIVE.
# ROAS surged from ~9.00 to ~15.00 (+66.7%) → POSITIVE.
# Two POSITIVE anomalies, zero CRITICALs.


# ──────────────────────────────────────────────────────────────────────
#  Case C — Stable Account
# ──────────────────────────────────────────────────────────────────────
# Scenario: A mature, well-optimised account with minor week-over-week
# noise. No metric crosses the configured thresholds.

case_c = WeeklyComparisonInput(
    account_id="acc_stable_237",
    client_name="GreenLeaf Landscaping (Stable)",
    period_current="2026-W29",
    period_previous="2026-W28",
    current_metrics=MetricSet(
        impressions=55_000,
        clicks=2_100,
        spend=3_400.00,
        conversions=68,
        revenue=8_500.00,
    ),
    previous_metrics=MetricSet(
        impressions=58_000,
        clicks=2_050,
        spend=3_250.00,
        conversions=71,
        revenue=8_100.00,
    ),
)

# Expected: CPA up ~+2.0%, ROAS up ~+4.6%, CTR roughly flat — none
# breach ±15% thresholds so total_anomalies_found should be 0.
# Severity = NEUTRAL if we wanted but with zero anomalies there's nothing.


# ──────────────────────────────────────────────────────────────────────
#  Registry for batch processing
# ──────────────────────────────────────────────────────────────────────

ALL_CASES: list[WeeklyComparisonInput] = [case_a, case_b, case_c]
