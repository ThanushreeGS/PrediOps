"""
report.py
---------
Generates an interactive, dynamic AIOps Operator Dashboard:
  - Simple Checkbox filter controls (Racks, Resources, Urgency, Regime Changes)
  - Clear Root Cause Diagnosis (driver tenant, slope acceleration, chronology)
  - Clear Prescriptive Recommended Decision (steps, deadline countdown, runway impact, runbook checklist)
  - Interactive Dynamic Vector Chart with hover inspection & What-If mitigation simulator
  - Model Validation against historical incidents (Success Metrics) & cryptographic Audit Trail

Zero external CDN dependencies: fully self-contained HTML5/CSS3/JavaScript.
"""

import json
from pathlib import Path
import pandas as pd


def _urgency_rank(result: dict) -> float:
    fc = result.get("forecast", {})
    if fc.get("status") != "ok":
        return float("inf")
    d = fc.get("to_safety_margin", {}).get("eta_days")
    return d if d is not None else float("inf")


def build_dynamic_report(pipeline_output: dict, backtest_output: dict, out_path: Path):
    results = sorted(pipeline_output.get("results", []), key=_urgency_rank)

    # Pre-calculate summary counters
    total_series = len(results)
    urgent_count = sum(1 for r in results if r.get("urgency", {}).get("code") == "URGENT")
    watch_count = sum(1 for r in results if r.get("urgency", {}).get("code") == "WATCH")
    healthy_count = sum(1 for r in results if r.get("urgency", {}).get("code") == "HEALTHY")
    regime_count = sum(len(r.get("change_points", [])) for r in results)

    # Unique racks and resources for simple checkboxes
    racks = sorted(list({r["rack_id"] for r in results}))
    resources = sorted(list({r["resource"] for r in results}))

    # JSON payloads safely encoded for direct embedding in the client-side script
    pipeline_json = json.dumps(results)
    backtest_json = json.dumps(backtest_output)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AIOps Capacity Exhaustion Forecaster & Decision Console</title>
<style>
  :root {{
    --bg: #f8fafc;
    --card: #ffffff;
    --card-hover: #f1f5f9;
    --text: #0f172a;
    --text-dim: #64748b;
    --border: #e2e8f0;
    --primary: #2563eb;
    --primary-bg: #eff6ff;
    --urgent: #dc2626;
    --urgent-bg: #fee2e2;
    --urgent-border: #fca5a5;
    --watch: #d97706;
    --watch-bg: #fef3c7;
    --watch-border: #fcd34d;
    --healthy: #16a34a;
    --healthy-bg: #dcfce7;
    --healthy-border: #86efac;
    --accent: #7c3aed;
    --accent-bg: #f5f3ff;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0b1120;
      --card: #131d33;
      --card-hover: #1b2845;
      --text: #f1f5f9;
      --text-dim: #94a3b8;
      --border: #23314d;
      --primary: #3b82f6;
      --primary-bg: #1e293b;
      --urgent: #ef4444;
      --urgent-bg: #451a1a;
      --urgent-border: #7f1d1d;
      --watch: #f59e0b;
      --watch-bg: #452c0f;
      --watch-border: #78350f;
      --healthy: #22c55e;
      --healthy-bg: #143522;
      --healthy-border: #14532d;
      --accent: #a855f7;
      --accent-bg: #2d1847;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    }}
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding-bottom: 60px;
  }}

  .container {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 20px;
  }}

  /* Top Navigation / Header */
  header {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}

  .header-left h1 {{
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .header-left .sub {{
    color: var(--text-dim);
    font-size: 0.88rem;
    margin-top: 4px;
  }}

  .badge-tag {{
    display: inline-block;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    border-radius: 6px;
    background: var(--primary-bg);
    color: var(--primary);
  }}

  /* Quick Summary Metric Cards */
  .metrics-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }}

  .metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    box-shadow: var(--shadow);
    text-align: left;
    transition: transform 0.15s ease;
  }}
  .metric-card:hover {{ transform: translateY(-2px); }}
  .metric-card .val {{ font-size: 1.7rem; font-weight: 800; line-height: 1.2; }}
  .metric-card .lbl {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-dim); font-weight: 600; margin-top: 4px; }}
  .metric-urgent {{ border-left: 4px solid var(--urgent); }}
  .metric-urgent .val {{ color: var(--urgent); }}
  .metric-watch {{ border-left: 4px solid var(--watch); }}
  .metric-watch .val {{ color: var(--watch); }}
  .metric-healthy {{ border-left: 4px solid var(--healthy); }}
  .metric-healthy .val {{ color: var(--healthy); }}
  .metric-accent {{ border-left: 4px solid var(--accent); }}
  .metric-accent .val {{ color: var(--accent); }}

  /* Filter Console with Simple Checkboxes */
  .filter-panel {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
  }}

  .filter-title {{
    font-size: 0.85rem;
    text-transform: uppercase;
    font-weight: 800;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .filter-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
  }}

  .filter-group {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}

  .group-label {{
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .checkbox-cluster {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}

  .cb-label {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    user-select: none;
    transition: all 0.15s ease;
  }}
  .cb-label:hover {{
    border-color: var(--primary);
    background: var(--primary-bg);
  }}
  .cb-label input[type="checkbox"] {{
    accent-color: var(--primary);
    width: 15px;
    height: 15px;
    cursor: pointer;
  }}
  .cb-label.active {{
    border-color: var(--primary);
    background: var(--primary-bg);
    font-weight: 600;
  }}

  .filter-btn-group {{
    display: flex;
    gap: 8px;
  }}
  .btn-sm {{
    padding: 4px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    cursor: pointer;
  }}
  .btn-sm:hover {{
    background: var(--primary-bg);
    border-color: var(--primary);
    color: var(--primary);
  }}

  /* Main Layout: Split into Matrix Table & Deep-Dive Dossier */
  .layout-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }}

  /* Capacity Urgency Table */
  .table-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }}

  .table-header-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    flex-wrap: wrap;
    gap: 10px;
  }}

  .table-header-row h2 {{
    font-size: 1.15rem;
    font-weight: 700;
  }}

  .table-wrap {{
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border);
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    text-align: left;
  }}

  thead th {{
    background: var(--bg);
    color: var(--text-dim);
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}

  tbody tr {{
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.15s ease;
  }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--card-hover); }}
  tbody tr.selected-row {{
    background: var(--primary-bg);
    outline: 2px solid var(--primary);
  }}

  td {{
    padding: 12px 14px;
    vertical-align: middle;
  }}

  /* Status Badges */
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    white-space: nowrap;
  }}
  .badge-red {{ background: var(--urgent-bg); color: var(--urgent); border: 1px solid var(--urgent-border); }}
  .badge-amber {{ background: var(--watch-bg); color: var(--watch); border: 1px solid var(--watch-border); }}
  .badge-green {{ background: var(--healthy-bg); color: var(--healthy); border: 1px solid var(--healthy-border); }}
  .badge-purple {{ background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent); }}
  .badge-grey {{ background: var(--border); color: var(--text-dim); }}

  /* Interactive Deep-Dive Dossier */
  .dossier-card {{
    background: var(--card);
    border: 2px solid var(--primary);
    border-radius: 14px;
    padding: 24px;
    box-shadow: var(--shadow-lg);
    margin-bottom: 24px;
    animation: fadeIn 0.25s ease-in-out;
  }}

  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  .dossier-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
  }}

  .dossier-title {{
    font-size: 1.3rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .dossier-grid {{
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 24px;
  }}
  @media (max-width: 950px) {{
    .dossier-grid {{ grid-template-columns: 1fr; }}
  }}

  /* Dynamic Chart Container */
  .chart-box {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}

  .chart-controls {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.78rem;
    color: var(--text-dim);
  }}

  .chart-toggles {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }}

  .svg-chart {{
    width: 100%;
    height: 280px;
    overflow: visible;
  }}

  /* Interactive What-If Simulator */
  .simulator-box {{
    background: var(--card);
    border: 1px dashed var(--primary);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 10px;
  }}
  .simulator-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 8px;
  }}
  .sim-slider-row {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .sim-slider-row input[type="range"] {{
    flex: 1;
    accent-color: var(--primary);
    cursor: pointer;
  }}
  .sim-feedback {{
    font-size: 0.82rem;
    margin-top: 6px;
    font-weight: 600;
    color: var(--healthy);
  }}

  /* Cause & Decision Cards */
  .intel-stack {{
    display: flex;
    flex-direction: column;
    gap: 18px;
  }}

  .cause-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 5px solid var(--urgent);
    border-radius: 12px;
    padding: 18px;
  }}

  .cause-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }}
  .cause-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: var(--urgent);
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .cause-summary-text {{
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.45;
    margin-bottom: 12px;
  }}

  /* Attribution Progress Bar */
  .driver-bars {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 10px;
  }}
  .driver-item {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 0.78rem;
  }}
  .driver-meta {{
    display: flex;
    justify-content: space-between;
    font-weight: 600;
  }}
  .bar-bg {{
    width: 100%;
    height: 7px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    background: var(--primary);
    border-radius: 4px;
  }}

  /* Decision Card */
  .decision-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 5px solid var(--healthy);
    border-radius: 12px;
    padding: 18px;
  }}

  .decision-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: var(--healthy);
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
  }}

  .decision-title-text {{
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 6px;
  }}

  .decision-desc {{
    font-size: 0.88rem;
    color: var(--text-dim);
    margin-bottom: 12px;
  }}

  .deadline-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--watch-bg);
    color: var(--watch);
    border: 1px solid var(--watch-border);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 12px;
  }}

  .steps-list {{
    margin: 0;
    padding-left: 18px;
    font-size: 0.84rem;
    line-height: 1.5;
    margin-bottom: 14px;
  }}
  .steps-list li {{ margin-bottom: 4px; }}

  /* Interactive Operator Runbook Checklist */
  .runbook-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .runbook-title {{
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
  }}
  .checklist-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    margin-bottom: 6px;
    cursor: pointer;
  }}
  .checklist-item input[type="checkbox"] {{
    accent-color: var(--healthy);
    width: 15px;
    height: 15px;
    cursor: pointer;
  }}
  .checklist-item.done {{
    text-decoration: line-through;
    color: var(--text-dim);
  }}

  /* Validation & Success Metrics */
  .validation-section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    box-shadow: var(--shadow);
    margin-top: 24px;
  }}
  .val-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin: 16px 0;
  }}
  .val-metric {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }}
  .val-metric .num {{ font-size: 1.6rem; font-weight: 800; color: var(--primary); }}
  .val-metric .desc {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-dim); font-weight: 600; }}

  /* Tooltip for chart */
  #chart-tooltip {{
    position: absolute;
    background: rgba(15, 23, 42, 0.92);
    color: #ffffff;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: monospace;
    pointer-events: none;
    display: none;
    z-index: 1000;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
  }}

  footer {{
    color: var(--text-dim);
    font-size: 0.8rem;
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div id="chart-tooltip"></div>

<div class="container">
  <!-- Header -->
  <header>
    <div class="header-left">
      <h1>
        <span>⚡ AIOps Capacity Forecaster</span>
        <span class="badge-tag">AO-02 Dynamic Console</span>
      </h1>
      <div class="sub">Seeing trouble before capacity runs out · Real-time Decision & Diagnostic Intelligence</div>
    </div>
    <div>
      <span class="badge badge-grey">Read-Only Telemetry Ingestion</span>
      <span class="badge badge-purple" style="margin-left: 6px;">Audit Verifiable</span>
    </div>
  </header>

  <!-- Metric Counters Strip -->
  <div class="metrics-strip">
    <div class="metric-card metric-urgent">
      <div class="val" id="cnt-urgent">{urgent_count}</div>
      <div class="lbl">Urgent Exhaustion Alerts (≤ 30d)</div>
    </div>
    <div class="metric-card metric-watch">
      <div class="val" id="cnt-watch">{watch_count}</div>
      <div class="lbl">Watch Planning Items (31–90d)</div>
    </div>
    <div class="metric-card metric-healthy">
      <div class="val" id="cnt-healthy">{healthy_count}</div>
      <div class="lbl">Healthy Resources (> 90d)</div>
    </div>
    <div class="metric-card metric-accent">
      <div class="val" id="cnt-regime">{regime_count}</div>
      <div class="lbl">Regime Shifts Detected</div>
    </div>
  </div>

  <!-- Simple Checkbox Filter Panel -->
  <div class="filter-panel">
    <div class="filter-title">
      <span>Dynamic Filter Console (Interactive Checkboxes)</span>
      <div class="filter-btn-group">
        <button class="btn-sm" onclick="filterMgr.selectAll()">Select All</button>
        <button class="btn-sm" onclick="filterMgr.selectUrgentOnly()">Urgent Only</button>
        <button class="btn-sm" onclick="filterMgr.resetDefaults()">Reset</button>
      </div>
    </div>

    <div class="filter-grid">
      <!-- Urgency Checkboxes -->
      <div class="filter-group">
        <div class="group-label">Urgency Level:</div>
        <div class="checkbox-cluster">
          <label class="cb-label active">
            <input type="checkbox" id="cb-urg-urgent" checked onchange="filterMgr.apply()"> 🔴 Urgent (≤ 30d)
          </label>
          <label class="cb-label active">
            <input type="checkbox" id="cb-urg-watch" checked onchange="filterMgr.apply()"> 🟡 Watch (31–90d)
          </label>
          <label class="cb-label active">
            <input type="checkbox" id="cb-urg-healthy" checked onchange="filterMgr.apply()"> 🟢 Healthy (> 90d)
          </label>
        </div>
      </div>

      <!-- Rack Checkboxes -->
      <div class="filter-group">
        <div class="group-label">Racks:</div>
        <div class="checkbox-cluster">
          {"".join(f'<label class="cb-label active"><input type="checkbox" class="cb-rack" value="{rk}" checked onchange="filterMgr.apply()"> {rk}</label>' for rk in racks)}
        </div>
      </div>

      <!-- Resource Checkboxes -->
      <div class="filter-group">
        <div class="group-label">Resources:</div>
        <div class="checkbox-cluster">
          {"".join(f'<label class="cb-label active"><input type="checkbox" class="cb-res" value="{rs}" checked onchange="filterMgr.apply()"> {rs}</label>' for rs in resources)}
        </div>
      </div>

      <!-- View & Special Toggles -->
      <div class="filter-group">
        <div class="group-label">Diagnostic Options:</div>
        <div class="checkbox-cluster">
          <label class="cb-label active">
            <input type="checkbox" id="cb-show-cause" checked onchange="filterMgr.toggleDetails()"> 🔍 Cause Dossier
          </label>
          <label class="cb-label active">
            <input type="checkbox" id="cb-show-decision" checked onchange="filterMgr.toggleDetails()"> 🎯 Decision Playbook
          </label>
          <label class="cb-label">
            <input type="checkbox" id="cb-only-regime" onchange="filterMgr.apply()"> ⚡ Only Regime Changes
          </label>
        </div>
      </div>
    </div>
  </div>

  <!-- Deep Dive Diagnostic & Decision Dossier (Selected Item) -->
  <div class="dossier-card" id="dossier-container">
    <div class="dossier-header">
      <div>
        <div class="dossier-title" id="dossier-title">
          <span>Loading series...</span>
        </div>
        <div style="font-size:0.85rem; color:var(--text-dim); margin-top:4px;" id="dossier-subtitle">
          Select any table row to inspect Root Cause & Recommended Operator Decision
        </div>
      </div>
      <div id="dossier-status-badge"></div>
    </div>

    <div class="dossier-grid">
      <!-- Left: Interactive Vector Chart + What-If Simulator -->
      <div class="chart-box">
        <div class="chart-controls">
          <span style="font-weight:700;">Telemetry Trend & Confidence Corridor</span>
          <div class="chart-toggles">
            <label><input type="checkbox" id="toggle-conf" checked onchange="chartMgr.render()"> 80% Band</label>
            <label><input type="checkbox" id="toggle-regimes" checked onchange="chartMgr.render()"> Regimes</label>
            <label><input type="checkbox" id="toggle-margins" checked onchange="chartMgr.render()"> Limits</label>
          </div>
        </div>

        <!-- SVG Vector Canvas -->
        <svg id="series-chart" class="svg-chart" viewBox="0 0 620 280"></svg>

        <!-- Live What-If Mitigation Simulator -->
        <div class="simulator-box">
          <div class="simulator-header">
            <span>⚙️ Live What-If Mitigation Simulator</span>
            <span id="sim-pct-label">0% Offload</span>
          </div>
          <div class="sim-slider-row">
            <span style="font-size:0.75rem; color:var(--text-dim);">No Action</span>
            <input type="range" id="sim-slider" min="0" max="80" step="5" value="0" oninput="simMgr.onSlide(this.value)">
            <span style="font-size:0.75rem; color:var(--text-dim);">-80% Load</span>
          </div>
          <div class="sim-feedback" id="sim-feedback-msg">
            Move slider to simulate rebalancing tenant workload away from this rack.
          </div>
        </div>
      </div>

      <!-- Right: Cause Analysis & Recommended Decision Stack -->
      <div class="intel-stack">
        <!-- Root Cause Dossier -->
        <div class="cause-card" id="cause-card-box">
          <div class="cause-header">
            <div class="cause-label">
              <span>🔍 Root Cause Diagnosis</span>
            </div>
            <span class="badge badge-purple" id="cause-cat-badge">Workload Scaling</span>
          </div>
          <div class="cause-summary-text" id="cause-summary-text">
            Analyzing telemetry attribution...
          </div>

          <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:var(--text-dim); margin-top:8px;">
            Workload Growth Attribution (% of net trend):
          </div>
          <div class="driver-bars" id="driver-bars-container">
            <!-- Dynamic Driver Bars -->
          </div>
        </div>

        <!-- Recommended Decision & Playbook -->
        <div class="decision-card" id="decision-card-box">
          <div class="decision-label">
            <span>🎯 Recommended Operator Decision</span>
          </div>
          <div class="decision-title-text" id="decision-title-text">
            Executing Decision Model...
          </div>
          <div class="deadline-badge" id="decision-deadline-badge">
            ⏱️ Action Target: Calculating...
          </div>
          <div class="decision-desc" id="decision-desc-text">
            Prescriptive guidance description.
          </div>

          <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:var(--text-dim); margin-bottom:6px;">
            Action Playbook & Operational Steps:
          </div>
          <ol class="steps-list" id="decision-steps-list">
            <!-- Steps list -->
          </ol>

          <!-- Interactive Runbook Checklist -->
          <div class="runbook-box">
            <div class="runbook-title">
              <span>Operator Execution Checklist</span>
              <span id="runbook-progress">0/4 Done</span>
            </div>
            <div id="runbook-items-container">
              <!-- Checkboxes for operator actions -->
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Facility Capacity Matrix Table -->
  <div class="table-card">
    <div class="table-header-row">
      <h2>Facility-Wide Urgency & Decision Matrix</h2>
      <div style="font-size:0.82rem; color:var(--text-dim);" id="table-filter-count">
        Showing all {total_series} series
      </div>
    </div>

    <div class="table-wrap">
      <table id="matrix-table">
        <thead>
          <tr>
            <th style="width:40px;"><input type="checkbox" id="th-select-all" onchange="tableMgr.toggleAllRows(this.checked)"></th>
            <th>Status</th>
            <th>Rack</th>
            <th>Resource</th>
            <th>Current Value</th>
            <th>Time to Safety Margin</th>
            <th>80% Range</th>
            <th>Root Cause</th>
            <th>Recommended Decision</th>
          </tr>
        </thead>
        <tbody id="matrix-tbody">
          <!-- Populated by JS -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- Validation & Model Audit -->
  <div class="validation-section">
    <h2 style="font-size:1.15rem; font-weight:700; margin-bottom:4px;">Historical Incident Validation (Success Metrics AO-02)</h2>
    <div style="font-size:0.85rem; color:var(--text-dim);">
      Verified via walk-forward backtest against held-out historical incident ground truth.
    </div>

    <div class="val-grid">
      <div class="val-metric">
        <div class="num" id="val-caught">--</div>
        <div class="desc">Incidents Foreseen in Advance</div>
      </div>
      <div class="val-metric">
        <div class="num" id="val-recall">--</div>
        <div class="desc">Regime Change Recall</div>
      </div>
      <div class="val-metric">
        <div class="num" id="val-fa">0%</div>
        <div class="desc">False Alarm Rate on Transient Spikes</div>
      </div>
    </div>

    <div style="font-size:0.82rem; color:var(--text-dim); line-height:1.5;">
      <b>Auditability Guarantee:</b> Every forecast computation logs an append-only entry to 
      <code>audit/forecast_audit_log.jsonl</code> with an immutable SHA-256 hash of its input data window.
      Read-only architecture guarantees no writes to cooling, power, or hypervisor controls.
    </div>
  </div>

  <footer>
    AIOps Capacity Forecasting Prototype · Cynefian Problem AO-02: Seeing trouble before capacity runs out.
  </footer>
</div>

<!-- Embedded Dataset -->
<script>
  const ALL_DATA = {pipeline_json};
  const BACKTEST_DATA = {backtest_json};

  let selectedIndex = 0;
  let simulatedReduction = 0;

  // Filter Manager
  const filterMgr = {{
    apply: function() {{
      const urgChecked = {{
        URGENT: document.getElementById('cb-urg-urgent').checked,
        WATCH: document.getElementById('cb-urg-watch').checked,
        HEALTHY: document.getElementById('cb-urg-healthy').checked
      }};

      const selectedRacks = new Set(
        Array.from(document.querySelectorAll('.cb-rack:checked')).map(el => el.value)
      );

      const selectedResources = new Set(
        Array.from(document.querySelectorAll('.cb-res:checked')).map(el => el.value)
      );

      const onlyRegime = document.getElementById('cb-only-regime').checked;

      const filtered = ALL_DATA.filter((r, idx) => {{
        const code = r.urgency ? r.urgency.code : 'HEALTHY';
        if (!urgChecked[code]) return false;
        if (!selectedRacks.has(r.rack_id)) return false;
        if (!selectedResources.has(r.resource)) return false;
        if (onlyRegime && (!r.change_points || r.change_points.length === 0)) return false;
        return true;
      }});

      tableMgr.render(filtered);
      document.getElementById('table-filter-count').innerText =
        `Showing ${{filtered.length}} of ${{ALL_DATA.length}} series`;

      // Update pill active classes
      document.querySelectorAll('.cb-label').forEach(lbl => {{
        const inp = lbl.querySelector('input[type="checkbox"]');
        if (inp) {{
          if (inp.checked) lbl.classList.add('active');
          else lbl.classList.remove('active');
        }}
      }});
    }},

    selectAll: function() {{
      document.querySelectorAll('.cb-rack, .cb-res, #cb-urg-urgent, #cb-urg-watch, #cb-urg-healthy')
        .forEach(cb => cb.checked = true);
      document.getElementById('cb-only-regime').checked = false;
      this.apply();
    }},

    selectUrgentOnly: function() {{
      document.getElementById('cb-urg-urgent').checked = true;
      document.getElementById('cb-urg-watch').checked = false;
      document.getElementById('cb-urg-healthy').checked = false;
      document.querySelectorAll('.cb-rack, .cb-res').forEach(cb => cb.checked = true);
      this.apply();
    }},

    resetDefaults: function() {{
      this.selectAll();
    }},

    toggleDetails: function() {{
      const showCause = document.getElementById('cb-show-cause').checked;
      const showDecision = document.getElementById('cb-show-decision').checked;
      document.getElementById('cause-card-box').style.display = showCause ? 'block' : 'none';
      document.getElementById('decision-card-box').style.display = showDecision ? 'block' : 'none';
    }}
  }};

  // Table Manager
  const tableMgr = {{
    currentList: [],

    render: function(list) {{
      this.currentList = list;
      const tbody = document.getElementById('matrix-tbody');
      tbody.innerHTML = '';

      if (list.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:24px; color:var(--text-dim);">No series matching selected checkboxes.</td></tr>';
        return;
      }}

      list.forEach((item, lIdx) => {{
        const globalIdx = ALL_DATA.indexOf(item);
        const tr = document.createElement('tr');
        if (globalIdx === selectedIndex) tr.classList.add('selected-row');

        tr.onclick = (e) => {{
          if (e.target.type === 'checkbox') return;
          dossierMgr.select(globalIdx);
        }};

        const urg = item.urgency || {{ code: 'HEALTHY', badge_class: 'badge-green', label: 'Healthy' }};
        const fc = item.forecast || {{}};
        const cause = item.cause_analysis || {{}};
        const dec = item.recommended_decision || {{}};

        let etaStr = "—";
        let rangeStr = "—";
        if (fc.status === 'ok' && fc.to_safety_margin && fc.to_safety_margin.eta_days !== null) {{
          etaStr = `<strong>${{Math.round(fc.to_safety_margin.eta_days)}} d</strong> <span style="color:var(--text-dim); font-size:0.75rem;">(${{fc.to_safety_margin.eta_date}})</span>`;
          const rng = fc.to_safety_margin.range_days;
          rangeStr = `${{Math.round(rng[0])}}–${{Math.round(rng[1])}} d`;
        }}

        const curVal = item.history && item.history.values.length ? item.history.values[item.history.values.length - 1] : "—";

        tr.innerHTML = `
          <td><input type="checkbox" class="row-select-cb" data-idx="${{globalIdx}}"></td>
          <td><span class="badge ${{urg.badge_class}}">${{urg.code}}</span></td>
          <td><strong>${{item.rack_id}}</strong></td>
          <td>${{item.resource}} <span style="color:var(--text-dim); font-size:0.75rem;">(${{item.unit}})</span></td>
          <td>${{curVal}} / <span style="color:var(--watch);">${{item.safety_margin}}</span></td>
          <td>${{etaStr}}</td>
          <td>${{rangeStr}}</td>
          <td>
            <div style="font-weight:600; font-size:0.8rem;">${{cause.title || 'Normal'}}</div>
            <div style="font-size:0.72rem; color:var(--text-dim);">${{cause.primary_driver || 'N/A'}} (${{cause.driver_share_pct || 0}}%)</div>
          </td>
          <td>
            <div style="font-weight:600; font-size:0.8rem; color:var(--primary);">${{dec.title || 'Monitor'}}</div>
            <div style="font-size:0.72rem; color:var(--text-dim);">${{dec.action_deadline_date ? 'By ' + dec.action_deadline_date : 'Routine'}}</div>
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }},

    toggleAllRows: function(checked) {{
      document.querySelectorAll('.row-select-cb').forEach(cb => cb.checked = checked);
    }}
  }};

  // Dossier & Inspector Manager
  const dossierMgr = {{
    select: function(idx) {{
      selectedIndex = idx;
      simulatedReduction = 0;
      document.getElementById('sim-slider').value = 0;
      document.getElementById('sim-pct-label').innerText = "0% Offload";
      document.getElementById('sim-feedback-msg').innerText = "Move slider to simulate rebalancing tenant workload away from this rack.";

      const item = ALL_DATA[idx];
      if (!item) return;

      // Update table selection highlight
      document.querySelectorAll('#matrix-tbody tr').forEach(tr => tr.classList.remove('selected-row'));
      filterMgr.apply();

      const urg = item.urgency || {{}};
      const cause = item.cause_analysis || {{}};
      const dec = item.recommended_decision || {{}};
      const fc = item.forecast || {{}};

      // Title & Header
      document.getElementById('dossier-title').innerHTML = `
        <span>${{item.rack_id}} — ${{item.resource}}</span>
        <span style="font-size:0.85rem; color:var(--text-dim); font-weight:500;">(Capacity Limit: ${{item.limit}} ${{item.unit}}, Safety Margin: ${{item.safety_margin}} ${{item.unit}})</span>
      `;
      document.getElementById('dossier-subtitle').innerText =
        `As of ${{fc.as_of_date || 'Latest'}} · Current Utilization: ${{cause.current_utilization || '—'}} ${{item.unit}}`;

      document.getElementById('dossier-status-badge').innerHTML = `
        <span class="badge ${{urg.badge_class}}" style="font-size:0.82rem; padding:5px 12px;">${{urg.label || urg.code}}</span>
      `;

      // Cause Card
      document.getElementById('cause-cat-badge').innerText = cause.category || 'Baseline';
      document.getElementById('cause-summary-text').innerText = cause.summary || 'Telemetry tracking normally within margins.';

      // Driver Bars
      const barsContainer = document.getElementById('driver-bars-container');
      barsContainer.innerHTML = '';
      if (item.attribution && item.attribution.length > 0) {{
        item.attribution.slice(0, 4).forEach(a => {{
          const pct = Math.round(a.share_of_rack_growth * 100);
          const div = document.createElement('div');
          div.className = 'driver-item';
          div.innerHTML = `
            <div class="driver-meta">
              <span>${{a.tenant_id}}</span>
              <span>${{pct}}% (${{a.slope_per_day}} ${{item.unit}}/day)</span>
            </div>
            <div class="bar-bg">
              <div class="bar-fill" style="width: ${{Math.min(100, pct)}}%;"></div>
            </div>
          `;
          barsContainer.appendChild(div);
        }});
      }} else {{
        barsContainer.innerHTML = '<div style="font-size:0.8rem; color:var(--text-dim);">Shared physical resource or evenly distributed baseline load.</div>';
      }}

      // Decision Card
      document.getElementById('decision-title-text').innerText = dec.title || 'Maintain Standard Monitoring';
      document.getElementById('decision-desc-text').innerText = dec.action_summary || 'No corrective action required.';
      document.getElementById('decision-deadline-badge').innerHTML = `
        ⏱️ Target Window: ${{dec.action_deadline_date ? dec.action_deadline_date + ' (' + (dec.action_deadline_days || 0) + 'd left)' : 'Routine Cycle'}}
      `;

      // Steps list
      const stepsList = document.getElementById('decision-steps-list');
      stepsList.innerHTML = '';
      if (dec.prescriptive_steps && dec.prescriptive_steps.length > 0) {{
        dec.prescriptive_steps.forEach(st => {{
          const li = document.createElement('li');
          li.innerText = st;
          stepsList.appendChild(li);
        }});
      }}

      // Runbook Checklist
      const checklistContainer = document.getElementById('runbook-items-container');
      checklistContainer.innerHTML = '';
      const tasks = dec.operator_checklist || ['Acknowledge capacity state'];
      tasks.forEach((t, tIdx) => {{
        const lbl = document.createElement('label');
        lbl.className = 'checklist-item';
        lbl.innerHTML = `<input type="checkbox" onchange="dossierMgr.toggleTask(this)"> <span>${{t}}</span>`;
        checklistContainer.appendChild(lbl);
      }});
      this.updateProgress();

      // Render Dynamic Chart
      chartMgr.render();
    }},

    toggleTask: function(checkbox) {{
      const span = checkbox.parentElement.querySelector('span');
      if (checkbox.checked) {{
        checkbox.parentElement.classList.add('done');
      }} else {{
        checkbox.parentElement.classList.remove('done');
      }}
      this.updateProgress();
    }},

    updateProgress: function() {{
      const total = document.querySelectorAll('.checklist-item input[type="checkbox"]').length;
      const done = document.querySelectorAll('.checklist-item input[type="checkbox"]:checked').length;
      document.getElementById('runbook-progress').innerText = `${{done}}/${{total}} Completed`;
    }}
  }};

  // Simulator Manager
  const simMgr = {{
    onSlide: function(val) {{
      simulatedReduction = parseInt(val, 10);
      document.getElementById('sim-pct-label').innerText = `${{simulatedReduction}}% Offload`;

      const item = ALL_DATA[selectedIndex];
      const fc = item ? item.forecast : null;
      if (!fc || fc.status !== 'ok' || !fc.to_safety_margin || fc.to_safety_margin.eta_days === null) {{
        document.getElementById('sim-feedback-msg').innerText = "Resource is already healthy with no positive exhaustion slope.";
        return;
      }}

      if (simulatedReduction === 0) {{
        document.getElementById('sim-feedback-msg').innerText = "Move slider to simulate rebalancing tenant workload away from this rack.";
      }} else {{
        const curEta = fc.to_safety_margin.eta_days;
        const newSlopeFactor = Math.max(0.05, 1 - (simulatedReduction / 100));
        const simEta = Math.round(curEta / newSlopeFactor);
        const gainedDays = simEta - Math.round(curEta);
        document.getElementById('sim-feedback-msg').innerHTML =
          `⚡ ${{simulatedReduction}}% reduction gains <strong>+${{gainedDays}} days</strong> of runway! (New ETA: ~${{simEta}} days).`;
      }}

      chartMgr.render();
    }}
  }};

  // SVG Chart Manager
  const chartMgr = {{
    render: function() {{
      const item = ALL_DATA[selectedIndex];
      if (!item) return;

      const svg = document.getElementById('series-chart');
      svg.innerHTML = '';

      const showBand = document.getElementById('toggle-conf').checked;
      const showRegimes = document.getElementById('toggle-regimes').checked;
      const showLimits = document.getElementById('toggle-margins').checked;

      const history = item.history || {{ dates: [], values: [] }};
      const dates = history.dates;
      const values = history.values;
      if (!values || values.length === 0) return;

      const fc = item.forecast || {{}};
      const limit = item.limit;
      const margin = item.safety_margin;

      // Coordinate scaling
      const width = 620;
      const height = 280;
      const padL = 45;
      const padR = 25;
      const padT = 20;
      const padB = 35;

      const plotW = width - padL - padR;
      const plotH = height - padT - padB;

      const minVal = Math.min(...values) * 0.92;
      const maxVal = Math.max(limit, Math.max(...values)) * 1.05;

      const scaleX = (i, total) => padL + (i / (total - 1)) * plotW;
      const scaleY = (v) => padT + plotH - ((v - minVal) / (maxVal - minVal)) * plotH;

      // Total forecast horizon points
      let forecastDays = 0;
      if (fc.status === 'ok' && fc.to_safety_margin && fc.to_safety_margin.eta_days) {{
        forecastDays = Math.min(180, Math.round(fc.to_safety_margin.eta_days));
      }}
      const totalPoints = values.length + (forecastDays > 0 ? forecastDays : 0);

      // Grid Lines
      const gridYCount = 4;
      for (let g = 0; g <= gridYCount; g++) {{
        const gVal = minVal + (g / gridYCount) * (maxVal - minVal);
        const yPos = scaleY(gVal);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", padL);
        line.setAttribute("x2", width - padR);
        line.setAttribute("y1", yPos);
        line.setAttribute("y2", yPos);
        line.setAttribute("stroke", "var(--border)");
        line.setAttribute("stroke-dasharray", "3,3");
        svg.appendChild(line);

        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", padL - 6);
        txt.setAttribute("y", yPos + 3);
        txt.setAttribute("text-anchor", "end");
        txt.setAttribute("font-size", "9");
        txt.setAttribute("fill", "var(--text-dim)");
        txt.textContent = Math.round(gVal);
        svg.appendChild(txt);
      }}

      // Limits (Safety Margin & Hard Limit)
      if (showLimits) {{
        // Hard Limit Line
        const yLimit = scaleY(limit);
        const lLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        lLine.setAttribute("x1", padL);
        lLine.setAttribute("x2", width - padR);
        lLine.setAttribute("y1", yLimit);
        lLine.setAttribute("y2", yLimit);
        lLine.setAttribute("stroke", "var(--urgent)");
        lLine.setAttribute("stroke-dasharray", "5,4");
        lLine.setAttribute("stroke-width", "1.3");
        svg.appendChild(lLine);

        // Safety Margin Line
        const yMargin = scaleY(margin);
        const mLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        mLine.setAttribute("x1", padL);
        mLine.setAttribute("x2", width - padR);
        mLine.setAttribute("y1", yMargin);
        mLine.setAttribute("y2", yMargin);
        mLine.setAttribute("stroke", "var(--watch)");
        mLine.setAttribute("stroke-dasharray", "4,4");
        mLine.setAttribute("stroke-width", "1.3");
        svg.appendChild(mLine);
      }}

      // Historical Path
      let pathD = "";
      values.forEach((v, i) => {{
        const x = scaleX(i, totalPoints);
        const y = scaleY(v);
        pathD += (i === 0 ? `M ${{x}} ${{y}}` : ` L ${{x}} ${{y}}`);
      }});

      const histPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
      histPath.setAttribute("d", pathD);
      histPath.setAttribute("fill", "none");
      histPath.setAttribute("stroke", "var(--primary)");
      histPath.setAttribute("stroke-width", "1.8");
      svg.appendChild(histPath);

      // Regime Change Markers
      if (showRegimes && item.change_points) {{
        item.change_points.forEach(cp => {{
          const dayIdx = cp.day_index || 0;
          if (dayIdx < values.length) {{
            const x = scaleX(dayIdx, totalPoints);
            const rLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
            rLine.setAttribute("x1", x);
            rLine.setAttribute("x2", x);
            rLine.setAttribute("y1", padT);
            rLine.setAttribute("y2", height - padB);
            rLine.setAttribute("stroke", "var(--accent)");
            rLine.setAttribute("stroke-dasharray", "2,2");
            rLine.setAttribute("stroke-width", "1.5");
            svg.appendChild(rLine);

            const rLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
            rLabel.setAttribute("x", x);
            rLabel.setAttribute("y", padT + 12);
            rLabel.setAttribute("text-anchor", "middle");
            rLabel.setAttribute("font-size", "8");
            rLabel.setAttribute("font-weight", "bold");
            rLabel.setAttribute("fill", "var(--accent)");
            rLabel.textContent = "Regime Δ";
            svg.appendChild(rLabel);
          }}
        }});
      }}

      // Forecast Line & 80% Confidence Corridor
      if (fc.status === 'ok' && fc.to_safety_margin && fc.to_safety_margin.eta_days) {{
        const lastX = scaleX(values.length - 1, totalPoints);
        const lastY = scaleY(values[values.length - 1]);

        let effectiveSlope = fc.daily_growth_rate || 0.01;
        if (simulatedReduction > 0) {{
          effectiveSlope *= Math.max(0.05, 1 - (simulatedReduction / 100));
        }}

        const projDays = forecastDays > 0 ? forecastDays : 60;
        const endX = scaleX(values.length - 1 + projDays, totalPoints);
        const endY = scaleY(values[values.length - 1] + effectiveSlope * projDays);

        // Confidence Corridor polygon
        if (showBand && fc.to_safety_margin.range_days) {{
          const loDays = fc.to_safety_margin.range_days[0];
          const hiDays = fc.to_safety_margin.range_days[1];
          const loX = scaleX(values.length - 1 + Math.min(loDays, totalPoints - values.length), totalPoints);
          const hiX = scaleX(values.length - 1 + Math.min(hiDays, totalPoints - values.length), totalPoints);

          const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          poly.setAttribute("points", `${{loX}},${{padT}} ${{hiX}},${{padT}} ${{hiX}},${{height - padB}} ${{loX}},${{height - padB}}`);
          poly.setAttribute("fill", "var(--primary)");
          poly.setAttribute("fill-opacity", "0.08");
          svg.appendChild(poly);
        }}

        // Forecast Line
        const fLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        fLine.setAttribute("x1", lastX);
        fLine.setAttribute("y1", lastY);
        fLine.setAttribute("x2", endX);
        fLine.setAttribute("y2", endY);
        fLine.setAttribute("stroke", simulatedReduction > 0 ? "var(--healthy)" : "var(--primary)");
        fLine.setAttribute("stroke-dasharray", "4,4");
        fLine.setAttribute("stroke-width", "2");
        svg.appendChild(fLine);
      }}

      // Axes X labels (Dates)
      const dateLabels = [dates[0], dates[Math.floor(dates.length / 2)], dates[dates.length - 1]];
      dateLabels.forEach((d, i) => {{
        const xPos = padL + (i / 2) * (scaleX(values.length - 1, totalPoints) - padL);
        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", xPos);
        txt.setAttribute("y", height - 12);
        txt.setAttribute("font-size", "9");
        txt.setAttribute("fill", "var(--text-dim)");
        txt.setAttribute("text-anchor", i === 0 ? "start" : (i === 2 ? "end" : "middle"));
        txt.textContent = d;
        svg.appendChild(txt);
      }});
    }}
  }};

  // Backtest Success Metrics Initializer
  function initBacktest() {{
    if (!BACKTEST_DATA || !BACKTEST_DATA.lead_time) return;
    const lt = BACKTEST_DATA.lead_time.per_incident || [];
    const cd = BACKTEST_DATA.change_detection || {{}};

    const caughtCount = lt.filter(r => r.lead_time_weeks !== null).length;
    document.getElementById('val-caught').innerText = `${{caughtCount}} / ${{lt.length}}`;

    const recallPct = cd.regime_change_recall !== undefined ? Math.round(cd.regime_change_recall * 100) + '%' : 'N/A';
    document.getElementById('val-recall').innerText = recallPct;

    const faPct = cd.false_alarm_rate_on_spikes !== undefined ? Math.round(cd.false_alarm_rate_on_spikes * 100) + '%' : '0%';
    document.getElementById('val-fa').innerText = faPct;
  }}

  // Startup initialization
  window.addEventListener('DOMContentLoaded', () => {{
    filterMgr.apply();
    dossierMgr.select(0);
    initBacktest();
  }});
</script>

</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    if not (base / "change_detection.py").exists():
        base = base.parent

    reports_dir = base / "reports"
    reports_dir.mkdir(exist_ok=True)

    pipeline_json_path = reports_dir / "pipeline_output.json"
    backtest_json_path = reports_dir / "backtest_output.json"

    # Auto-run pipeline if output not found
    if not pipeline_json_path.exists():
        print("pipeline_output.json not found. Running pipeline...")
        from pipeline import run_pipeline
        data_dir = base / "data"
        if not (data_dir / "telemetry.csv").exists():
            from data_generator import generate_telemetry
            generate_telemetry(data_dir)
        pipe_out = run_pipeline(data_dir, base / "audit")
        with open(pipeline_json_path, "w") as f:
            json.dump(pipe_out, f, indent=2)

    # Auto-run backtest if output not found
    if not backtest_json_path.exists():
        print("backtest_output.json not found. Running backtest...")
        from backtest import backtest_lead_time, backtest_change_detection
        data_dir = base / "data"
        lt = backtest_lead_time(data_dir)
        cd = backtest_change_detection(data_dir)
        back_out = {"lead_time": lt, "change_detection": cd}
        with open(backtest_json_path, "w") as f:
            json.dump(back_out, f, indent=2, default=str)

    with open(pipeline_json_path, encoding="utf-8") as f:
        pipe_data = json.load(f)
    with open(backtest_json_path, encoding="utf-8") as f:
        back_data = json.load(f)

    out_file = reports_dir / "operator_report.html"
    build_dynamic_report(pipe_data, back_data, out_file)
    print("=" * 60)
    print("SUCCESS: Dynamic AIOps Operator Dashboard generated at:")
    print(out_file)
    print("=" * 60)
