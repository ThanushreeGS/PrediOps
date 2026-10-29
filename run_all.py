"""One command: data -> forecasts + cause + decision -> backtest -> operator console.
Usage:  python run_all.py
Read-only with respect to any facility system; writes only inside this folder."""
import json
from pathlib import Path

from data_generator import generate_telemetry
from pipeline import run_pipeline
import backtest
from build_console import build_console

base = Path(__file__).resolve().parent
data_dir, reports = base / "data", base / "reports"
reports.mkdir(exist_ok=True)

if not (data_dir / "telemetry.csv").exists():
    print("No data found, generating synthetic telemetry...")
    generate_telemetry(data_dir)

out = run_pipeline(data_dir, base / "audit")
(reports / "pipeline_output.json").write_text(json.dumps(out, indent=2))
print(f"Pipeline: {len(out['results'])} rack/resource series")
for r in out["results"]:
    u = r["urgency"]["code"]
    if u != "HEALTHY":
        print(f"  [{u}] {r['rack_id']} {r['resource']}: {r['recommended_decision']['title']}")

lt = backtest.backtest_lead_time(data_dir)
cd = backtest.backtest_change_detection(data_dir)
(reports / "backtest_output.json").write_text(json.dumps({"lead_time": lt, "change_detection": cd}, indent=2, default=str))
print(f"Backtest: recall={cd['regime_change_recall']}, false alarms={cd['false_alarm_rate_on_spikes']}")

print("Wrote", build_console())
print("Open reports/capacity_console.html in a browser.")
