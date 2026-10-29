"""
pipeline.py
-----------
Ties the modules together into the end-to-end flow described in the
problem statement's Mission: for every (rack, resource) pair, produce
an honest, precise, explained time-to-exhaustion forecast.

NOTE ON SCOPE (Evaluation Boundary): this pipeline only ingests
telemetry, forecasts, detects change, attributes cause, and produces
operator-facing output. It performs NO writes to any facility system -
every function here is pure/read-only, matching "the prototype reads
telemetry only; it must not change facility controls."
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from change_detection import detect_change_points
from forecasting import forecast_time_to_exhaustion, _current_regime_window
from attribution import attribute_rack_trend
from audit import log_forecast, _hash_values
from data_generator import RACK_SHARED_RESOURCES
from decision_engine import analyze_cause_and_decision


def run_pipeline(data_dir: Path, audit_dir: Path) -> dict:
    telemetry = pd.read_csv(data_dir / "telemetry.csv", parse_dates=["date"])
    limits = pd.read_csv(data_dir / "capacity_limits.csv").set_index("resource")

    results = []

    for (rack_id, resource), grp in telemetry.groupby(["rack_id", "resource"]):
        # thermal is a shared physical property -> average across sensors/
        # tenant-views rather than summing; everything else is additive.
        agg_fn = "mean" if resource in RACK_SHARED_RESOURCES else "sum"
        rack_agg = (grp.groupby("date")["value"].agg(agg_fn)
                    .sort_index().reset_index())
        dates = rack_agg["date"]
        values = rack_agg["value"].to_numpy()

        change_points = detect_change_points(dates, values)
        limit_row = limits.loc[resource]
        forecast = forecast_time_to_exhaustion(
            dates, values, float(limit_row["limit"]), float(limit_row["safety_margin"]),
            change_points,
        )

        # attribution needs each tenant's own series on this rack, aligned
        # to the same date index as the rack aggregate
        tenant_series = {}
        rack_tenants = grp["tenant_id"].unique()
        for tenant_id in rack_tenants:
            t_grp = (grp[grp["tenant_id"] == tenant_id]
                     .set_index("date")["value"].reindex(dates).fillna(0).to_numpy())
            tenant_series[tenant_id] = t_grp

        idx, _ = _current_regime_window(dates, values, change_points)
        attribution = attribute_rack_trend(tenant_series, int(idx[0]))

        data_window = {
            "start_date": str(dates.iloc[0].date()),
            "end_date": str(dates.iloc[-1].date()),
            "n_points": len(values),
            "data_hash": _hash_values(values),
        }
        log_forecast(audit_dir, rack_id, resource, data_window, forecast,
                      change_points, attribution)

        # AIOps Decision & Cause Intelligence Layer
        insights = analyze_cause_and_decision(
            rack_id=rack_id,
            resource=resource,
            unit=limit_row["unit"],
            limit=float(limit_row["limit"]),
            safety_margin=float(limit_row["safety_margin"]),
            forecast=forecast,
            change_points=change_points,
            attribution=attribution,
            current_value=float(values[-1])
        )

        results.append({
            "rack_id": rack_id,
            "resource": resource,
            "unit": limit_row["unit"],
            "limit": float(limit_row["limit"]),
            "safety_margin": float(limit_row["safety_margin"]),
            "change_points": change_points,
            "forecast": forecast,
            "attribution": attribution,
            "urgency": insights["urgency"],
            "cause_analysis": insights["cause_analysis"],
            "recommended_decision": insights["recommended_decision"],
            "history": {
                "dates": [str(d.date()) for d in dates],
                "values": [round(float(v), 2) for v in values],
            },
        })

    return {"results": results}


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    if not (base / "change_detection.py").exists():
        base = base.parent
    data_dir = base / "data"
    if not (data_dir / "telemetry.csv").exists():
        print("Data files not found. Generating synthetic telemetry in", data_dir)
        from data_generator import generate_telemetry
        generate_telemetry(data_dir)
    out = run_pipeline(data_dir, base / "audit")
    (base / "reports").mkdir(exist_ok=True)
    with open(base / "reports" / "pipeline_output.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Processed {len(out['results'])} rack/resource series.")
    for r in out["results"]:
        f = r["forecast"]
        u = r["urgency"]
        c = r["cause_analysis"]
        d = r["recommended_decision"]
        if f["status"] == "ok" and f["to_safety_margin"]["eta_days"] is not None:
            print(f"  [{u['code']:7s}] {r['rack_id']:10s} {r['resource']:14s} -> {f['to_safety_margin']['eta_days']:.0f}d to margin "
                  f"| Cause: {c['category']} ({c['primary_driver']}) "
                  f"| Decision: {d['title']}")
        else:
            print(f"  [{u['code']:7s}] {r['rack_id']:10s} {r['resource']:14s} -> stable baseline")
