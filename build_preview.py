"""
NarrateKPI — Build Preview Dashboard

Generates a standalone, polished HTML page that displays all three mock
reports (Cases A, B, C) in an interactive, professional dashboard.
Used to register a live preview for the Freebuff Desktop Preview tab.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure we can import sibling modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_engine import generate_report
from math_engine import run_math_engine
from mock_data import ALL_CASES

OUTPUT_HTML = "preview_report.html"


def _build_metrics_table(cur: Dict[str, Any], prev: Dict[str, Any]) -> str:
    """Build an HTML metrics comparison table."""
    metric_order = [
        ("Impressions", "impressions", "int"),
        ("Clicks", "clicks", "int"),
        ("Spend ($)", "spend", "float"),
        ("Conversions", "conversions", "int"),
        ("Revenue ($)", "revenue", "float"),
        ("CTR (%)", "ctr", "float"),
        ("CPC ($)", "cpc", "float"),
        ("CPA ($)", "cpa", "float"),
        ("ROAS (×)", "roas", "float"),
    ]

    rows_html = ""
    for label, key, typ in metric_order:
        c = cur.get(key, 0)
        p = prev.get(key, 0)
        if isinstance(c, (int, float)) and isinstance(p, (int, float)) and p:
            pct = ((c - p) / p) * 100
            arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "─")
            color = "#ef4444" if pct > 0 else "#22c55e" if pct < 0 else "#6b7280"
        else:
            pct = 0.0
            arrow = "─"
            color = "#6b7280"

        if typ == "int":
            c_str, p_str = f"{c:,}", f"{p:,}"
        else:
            c_str, p_str = f"{c:,.2f}", f"{p:,.2f}"

        rows_html += f"""<tr>
            <td style="font-weight:500">{label}</td>
            <td style="text-align:right;font-variant-numeric:tabular-nums">{c_str}</td>
            <td style="text-align:right;font-variant-numeric:tabular-nums">{p_str}</td>
            <td style="text-align:right;color:{color};font-weight:600;font-variant-numeric:tabular-nums">
                {arrow} {pct:+.2f}%
            </td>
        </tr>"""

    return f"""<table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
            <tr style="border-bottom:2px solid #e5e7eb;color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:0.05em">
                <th style="text-align:left;padding:8px 12px">Metric</th>
                <th style="text-align:right;padding:8px 12px">Current</th>
                <th style="text-align:right;padding:8px 12px">Previous</th>
                <th style="text-align:right;padding:8px 12px">Change</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>"""


def _build_anomaly_badge(severity: str) -> str:
    colors = {
        "CRITICAL": ("#fef2f2", "#dc2626", "🚨"),
        "WARNING": ("#fffbeb", "#d97706", "⚠️"),
        "POSITIVE": ("#f0fdf4", "#16a34a", "✅"),
        "NEUTRAL": ("#f9fafb", "#6b7280", "ℹ️"),
    }
    bg, txt, icon = colors.get(severity, ("#f9fafb", "#6b7280", "ℹ️"))
    return f"""<span style="display:inline-flex;align-items:center;gap:4px;
        padding:2px 10px;border-radius:9999px;font-size:11px;font-weight:600;
        background:{bg};color:{txt}">{icon} {severity}</span>"""


def _build_anomaly_card(anomaly: Dict[str, Any]) -> str:
    metric = anomaly.get("metric_name", "")
    sev = anomaly.get("severity", "")
    cur_val = anomaly.get("current_value", "")
    prev_val = anomaly.get("previous_value", "")
    pct = anomaly.get("percent_change", 0.0)
    msg = anomaly.get("message", "")

    return f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;margin-bottom:10px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-weight:600;font-size:14px">{metric}</span>
            {_build_anomaly_badge(sev)}
        </div>
        <div style="display:flex;gap:20px;margin-bottom:8px;font-size:12px;color:#6b7280">
            <span><strong>{cur_val}</strong> <span style="color:#9ca3af">(current)</span></span>
            <span><strong>{prev_val}</strong> <span style="color:#9ca3af">(previous)</span></span>
            <span style="color:{'#ef4444' if pct > 0 else '#22c55e'};font-weight:600">{pct:+.2f}%</span>
        </div>
        <div style="font-size:13px;color:#374151;line-height:1.5">{msg}</div>
    </div>"""


def _generate_markdown_html(markdown_text: str) -> str:
    """Convert simple Markdown to HTML for embedding."""
    html = markdown_text
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    # Italic
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    # Inline code
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    # Headings
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    # Blockquote
    html = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)
    # Horizontal rule
    html = html.replace("---", "<hr style='border:none;border-top:2px solid #e5e7eb;margin:16px 0'>")
    # Paragraphs (double newline)
    html = re.sub(r"\n\n+", "</p><p>", html)
    html = f"<p>{html}</p>"
    # Single newlines -> <br>
    html = re.sub(r"\n", "<br>", html)
    # Fix broken wrapping
    html = html.replace("</p><br>", "</p>")
    html = html.replace("<br></p>", "</p>")
    html = html.replace("</p><p><br>", "</p><p>")
    return html


def build_dashboard() -> str:
    """Run pipeline and generate the full HTML dashboard."""
    reports_data: List[Dict[str, Any]] = []

    for idx, case in enumerate(ALL_CASES):
        report = run_math_engine(case)
        summary = report.summary_raw_json
        markdown = generate_report(summary)

        cur = summary.get("current_metrics", {})
        prev = summary.get("previous_metrics", {})
        anomalies = summary.get("anomalies", [])
        metrics_html = _build_metrics_table(cur, prev)
        anomaly_cards = "\n".join(_build_anomaly_card(a) for a in anomalies)
        if not anomalies:
            anomaly_cards = (
                '<div style="text-align:center;padding:24px;color:#6b7280;'
                'font-size:14px;background:#f9fafb;border-radius:10px;'
                'border:1px dashed #d1d5db">'
                '📊 No anomalies detected — account is stable.</div>'
            )

        markdown_html = _generate_markdown_html(markdown)

        total = summary.get("total_anomalies_found", 0)
        sev_classes = ""
        if total > 0:
            crit = sum(1 for a in anomalies if a.get("severity") == "CRITICAL")
            pos = sum(1 for a in anomalies if a.get("severity") == "POSITIVE")
            warn = sum(1 for a in anomalies if a.get("severity") == "WARNING")
            if crit:
                sev_classes = "status-critical"
            elif warn:
                sev_classes = "status-warning"
            elif pos:
                sev_classes = "status-positive"
            else:
                sev_classes = "status-neutral"

        reports_data.append({
            "idx": idx + 1,
            "account_id": summary.get("account_id", ""),
            "client_name": summary.get("client_name", f"Case #{idx + 1}"),
            "period_current": summary.get("period_current", ""),
            "period_previous": summary.get("period_previous", ""),
            "total_anomalies": total,
            "sev_classes": sev_classes,
            "metrics_html": metrics_html,
            "anomaly_cards": anomaly_cards,
            "markdown_html": markdown_html,
            "summary_raw_json": summary,
        })

    # Generate navigation tabs for each report.
    nav_tabs = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f'onclick="switchTab({i})" data-index="{i}">'
        f'<span class="tab-indicator {r["sev_classes"]}"></span>'
        f'{r["client_name"].split(" (")[0]}'
        f'<span class="tab-count">{r["total_anomalies"]}</span>'
        f'</button>'
        for i, r in enumerate(reports_data)
    )

    # Build each tab panel.
    tab_panels = ""
    for i, r in enumerate(reports_data):
        active = " active" if i == 0 else ""
        tab_panels += f"""
        <div class="tab-panel{active}" id="panel-{i}">
            <div class="report-header">
                <div>
                    <div class="report-client">{r['client_name']}</div>
                    <div class="report-meta">
                        <span>📅 {r['period_current']} → {r['period_previous']}</span>
                        <span class="meta-sep">·</span>
                        <span>🆔 {r['account_id']}</span>
                    </div>
                </div>
                <div class="anomaly-count {r['sev_classes']}">
                    <span class="anomaly-num">{r['total_anomalies']}</span>
                    anomal{'ies' if r['total_anomalies'] != 1 else 'y'}
                </div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px">
                <div class="card">
                    <div class="card-title">📈 Metrics Comparison</div>
                    {r['metrics_html']}
                </div>
                <div class="card">
                    <div class="card-title">🔍 Anomaly Breakdown</div>
                    {r['anomaly_cards']}
                </div>
            </div>

            <div class="card" style="margin-bottom:24px">
                <div class="card-title">📝 Generated Report</div>
                <div class="markdown-body">{r['markdown_html']}</div>
            </div>

            <details style="margin-bottom:24px">
                <summary style="cursor:pointer;font-size:13px;color:#6b7280;padding:8px 0;font-weight:500">
                    📄 View Raw JSON Payload
                </summary>
                <pre style="background:#1e293b;color:#e2e8f0;padding:16px;border-radius:8px;overflow:auto;font-size:12px;margin-top:8px"><code>{json.dumps(r['summary_raw_json'], indent=2, ensure_ascii=False)}</code></pre>
            </details>
        </div>"""

    # Full HTML page.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NarrateKPI — Weekly Performance Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif; background:#f3f4f6; color:#111827; }}
  .app {{ max-width:1280px; margin:0 auto; padding:24px 32px; }}

  /* Header */
  .topbar {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; }}
  .logo {{ display:flex; align-items:center; gap:12px; }}
  .logo-icon {{ width:40px; height:40px; background:linear-gradient(135deg,#6366f1,#8b5cf6); border-radius:10px; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:18px; }}
  .logo-text {{ font-size:20px; font-weight:700; }}
  .logo-sub {{ font-size:12px; color:#6b7280; margin-top:-2px; }}
  .badge {{ display:inline-flex; align-items:center; gap:4px; padding:4px 12px; border-radius:9999px; font-size:11px; font-weight:600; background:#e0e7ff; color:#4338ca; }}

  /* Tabs */
  .tab-bar {{ display:flex; gap:4px; background:#fff; border-radius:12px; padding:4px; margin-bottom:20px; border:1px solid #e5e7eb; overflow-x:auto; }}
  .tab-btn {{ display:flex; align-items:center; gap:8px; padding:10px 18px; border:none; background:transparent; border-radius:8px; cursor:pointer; font-size:13px; font-weight:500; color:#6b7280; white-space:nowrap; transition:all 0.15s ease; }}
  .tab-btn:hover {{ background:#f3f4f6; color:#374151; }}
  .tab-btn.active {{ background:#6366f1; color:#fff; box-shadow:0 2px 8px rgba(99,102,241,0.3); }}
  .tab-indicator {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
  .tab-indicator.status-critical {{ background:#ef4444; }}
  .tab-indicator.status-warning {{ background:#f59e0b; }}
  .tab-indicator.status-positive {{ background:#22c55e; }}
  .tab-indicator.status-neutral {{ background:#9ca3af; }}
  .tab-count {{ display:inline-flex; align-items:center; justify-content:center; min-width:20px; height:20px; border-radius:9999px; font-size:11px; font-weight:600; padding:0 6px; background:rgba(255,255,255,0.2); }}
  .tab-btn.active .tab-count {{ background:rgba(255,255,255,0.25); }}

  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; animation:fadeIn 0.2s ease; }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}

  /* Report header */
  .report-header {{ display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:20px; gap:16px; }}
  .report-client {{ font-size:18px; font-weight:700; }}
  .report-meta {{ display:flex; align-items:center; gap:6px; font-size:13px; color:#6b7280; margin-top:4px; }}
  .meta-sep {{ color:#d1d5db; }}
  .anomaly-count {{ display:flex; align-items:center; gap:8px; padding:8px 20px; border-radius:10px; font-size:13px; font-weight:500; }}
  .anomaly-count.status-critical {{ background:#fef2f2; color:#dc2626; }}
  .anomaly-count.status-warning {{ background:#fffbeb; color:#d97706; }}
  .anomaly-count.status-positive {{ background:#f0fdf4; color:#16a34a; }}
  .anomaly-count.status-neutral {{ background:#f9fafb; color:#6b7280; }}
  .anomaly-num {{ font-size:24px; font-weight:700; }}

  /* Cards */
  .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:20px; }}
  .card-title {{ font-size:14px; font-weight:600; margin-bottom:14px; color:#374151; }}

  /* Markdown body */
  .markdown-body {{ font-size:13px; line-height:1.7; color:#374151; }}
  .markdown-body h2 {{ font-size:16px; font-weight:700; margin:20px 0 10px; color:#111827; }}
  .markdown-body h3 {{ font-size:14px; font-weight:600; margin:14px 0 8px; color:#1f2937; }}
  .markdown-body p {{ margin-bottom:10px; }}
  .markdown-body hr {{ margin:16px 0; border:none; border-top:2px solid #e5e7eb; }}
  .markdown-body strong {{ color:#111827; }}
  .markdown-body blockquote {{ border-left:3px solid #6366f1; padding:8px 14px; margin:10px 0; background:#f8f8ff; border-radius:0 6px 6px 0; font-size:12px; color:#6b7280; }}
  .markdown-body code {{ background:#f3f4f6; padding:2px 6px; border-radius:4px; font-size:12px; color:#6366f1; }}

  /* Responsive */
  @media (max-width:800px) {{ .app {{ padding:16px; }} .tab-btn {{ padding:8px 12px; font-size:12px; }} }}
  @media (max-width:900px) {{ .report-header {{ flex-direction:column; }} div[style*="grid-template-columns:1fr 1fr"] {{ grid-template-columns:1fr !important; }} }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width:6px; height:6px; }}
  ::-webkit-scrollbar-track {{ background:transparent; }}
  ::-webkit-scrollbar-thumb {{ background:#d1d5db; border-radius:3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background:#9ca3af; }}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="logo">
      <div class="logo-icon">NK</div>
      <div>
        <div class="logo-text">NarrateKPI</div>
        <div class="logo-sub">AI-Driven Agency Report Automation</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <span class="badge">📊 Week {reports_data[0]['period_current']}</span>
      <span class="badge">🔬 Dry-Run Mode</span>
    </div>
  </div>

  <div class="tab-bar">{nav_tabs}</div>
  {tab_panels}
</div>

<script>
function switchTab(idx) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab-btn[data-index="${{idx}}"]`).classList.add('active');
  document.getElementById(`panel-${{idx}}`).classList.add('active');
}}

// Keyboard shortcuts for tabs.
document.addEventListener('keydown', (e) => {{
  const num = parseInt(e.key);
  if (num >= 1 && num <= 3) {{
    switchTab(num - 1);
  }}
}});
</script>
</body>
</html>"""


def main() -> None:
    print("[NarrateKPI] 📊 Generating preview dashboard...")
    html = build_dashboard()
    output_path = Path(OUTPUT_HTML)
    output_path.write_text(html, encoding="utf-8")
    print(f"[NarrateKPI] ✅ Dashboard written → {output_path.resolve()}")
    print(f"[NarrateKPI] 🌐 Open in browser or register as preview.")


if __name__ == "__main__":
    main()
