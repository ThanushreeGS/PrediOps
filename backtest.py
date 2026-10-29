"""
backtest.py
-----------
Validates the pipeline against the hackathon's own Success Metrics:

  - Forecast accuracy on held-out periods, especially known past events
  - Lead time: how many weeks' warning was given before each historical
    exhaustion event
  - False-alarm rate: spikes that resolved on their own should NOT be
    reported as regime changes
  - Regime-change recall: did we catch the injected onboarding / policy
    changes, and roughly on time?

Method: walks forward in time. At each of several "as-of" checkpoints
(e.g. every 30 days), re-runs change-detection + forecasting using only
data available UP TO that checkpoint (no peeking at the future), then
checks whether/when the real incident_log events were foreseen, and how
many of the known injected regime changes vs spikes were reported as
which type.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from change_detection import detect_change_points
from forecasting import forecast_time_to_exhaustion, _current_regime_window
from data_generator import RACK_SHARED_RESOURCES

CHECKPOINT_STEP_DAYS = 21
MIN_HISTORY_DAYS = 60


def _rack_series(telemetry: pd.DataFrame, rack_id: str, resource: str) -> tuple[pd.Series, np.ndarray]:
    grp = telemetry[(telemetry.rack_id == rack_id) & (telemetry.resource == resource)]
    agg_fn = "mean" if resource in RACK_SHARED_RESOURCES else "sum"
    rack_agg = grp.groupby("date")["value"].agg(agg_fn).sort_index().reset_index()
    return rack_agg["date"], rack_agg["value"].to_numpy()


def backtest_lead_time(data_dir: Path) -> dict:
    telemetry = pd.read_csv(data_dir / "telemetry.csv", parse_dates=["date"])
    limits = pd.read_csv(data_dir / "capacity_limits.csv").set_index("resource")
    incidents = pd.read_csv(data_dir / "incident_log.csv", parse_dates=["event_date"])
    ground_truth = pd.read_csv(data_dir / "ground_truth.csv", parse_dates=["date"])

    lead_time_results = []

    for _, inc in incidents.iterrows():
        rack_id, resource = inc["rack_id"], inc["resource"]
        dates, values = _rack_series(telemetry, rack_id, resource)
        event_date = inc["event_date"]
        limit_row = limits.loc[resource]

        # walk forward from MIN_HISTORY_DAYS up to the event, checkpointing
        # every CHECKPOINT_STEP_DAYS, and record the FIRST checkpoint at
        # which the forecast's point estimate correctly foresaw the event
        # within its own 80% confidence range.
        first_correct_warning_date = None
        n = len(dates)
        for cp_end in range(MIN_HISTORY_DAYS, n, CHECKPOINT_STEP_DAYS):
            if dates.iloc[cp_end] >= event_date:
                break
            d_slice = dates.iloc[:cp_end]
            v_slice = values[:cp_end]
            cps = detect_change_points(d_slice, v_slice)
            fc = forecast_time_to_exhaustion(d_slice, v_slice, float(limit_row["limit"]),
                                              float(limit_row["safety_margin"]), cps)
            if fc["status"] != "ok":
                continue
            eta = fc["to_safety_margin"]["eta_date"]
            if eta is None:
                continue
            predicted_date = pd.Timestamp(eta)
            # "correct" = predicted exhaustion lands within +/- 25% of the
            # actual remaining time, or the actual event falls inside the
            # reported 80% range
            lo, hi = fc["to_safety_margin"]["range_dates"]
            within_range = pd.Timestamp(lo) <= event_date <= pd.Timestamp(hi)
            close_enough = abs((predicted_date - event_date).days) <= max(14, int(0.25 * (event_date - d_slice.iloc[-1]).days))
            if within_range or close_enough:
                first_correct_warning_date = d_slice.iloc[-1]
                break

        if first_correct_warning_date is not None:
            lead_days = (event_date - first_correct_warning_date).days
            lead_time_results.append({
                "rack_id": rack_id, "resource": resource,
                "event_date": str(event_date.date()),
                "first_correct_warning_date": str(first_correct_warning_date.date()),
                "lead_time_days": lead_days,
                "lead_time_weeks": round(lead_days / 7, 1),
            })
        else:
            lead_time_results.append({
                "rack_id": rack_id, "resource": resource,
                "event_date": str(event_date.date()),
                "first_correct_warning_date": None,
                "lead_time_days": None,
                "lead_time_weeks": None,
            })

    return {"per_incident": lead_time_results}


def backtest_change_detection(data_dir: Path) -> dict:
    """Checks recall/false-alarm behaviour against the injected ground truth:
    did we flag the real regime changes, and did we avoid flagging the
    transient spikes as regime changes?"""
    telemetry = pd.read_csv(data_dir / "telemetry.csv", parse_dates=["date"])
    ground_truth = pd.read_csv(data_dir / "ground_truth.csv", parse_dates=["date"])

    rows = []
    for (rack_id, resource), grp in telemetry.groupby(["rack_id", "resource"]):
        agg_fn = "mean" if resource in RACK_SHARED_RESOURCES else "sum"
        rack_agg = grp.groupby("date")["value"].agg(agg_fn).sort_index().reset_index()
        dates, values = rack_agg["date"], rack_agg["value"].to_numpy()
        detected = detect_change_points(dates, values)
        detected_days = [d["day_index"] for d in detected]

        truth_here = ground_truth[(ground_truth.rack_id == rack_id) & (ground_truth.resource == resource)]
        for _, ev in truth_here.iterrows():
            # for a tenant-level event on a multi-tenant rack, the rack
            # aggregate shift is softened, so allow a wider tolerance window
            matched = any(abs(dd - ev["day_index"]) <= 20 for dd in detected_days)
            rows.append({
                "rack_id": rack_id, "resource": resource,
                "event_type": ev["event_type"], "event_date": str(ev["date"]),
                "detected_by_model": bool(matched) if ev["event_type"] == "regime_change" else None,
                "correctly_ignored": (not matched) if ev["event_type"] == "transient_spike" else None,
            })

    df = pd.DataFrame(rows)
    regime = df[df.event_type == "regime_change"]
    spikes = df[df.event_type == "transient_spike"]
    recall = regime["detected_by_model"].mean() if len(regime) else None
    false_alarm_rate = 1 - spikes["correctly_ignored"].mean() if len(spikes) else None

    return {
        "regime_change_recall": None if recall is None else round(float(recall), 2),
        "n_regime_changes": int(len(regime)),
        "false_alarm_rate_on_spikes": None if false_alarm_rate is None else round(float(false_alarm_rate), 2),
        "n_transient_spikes": int(len(spikes)),
        "details": rows,
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    if not (base / "change_detection.py").exists():
        base = base.parent
    data_dir = base / "data"
    if not (data_dir / "telemetry.csv").exists():
        from data_generator import generate_telemetry
        generate_telemetry(data_dir)
    lt = backtest_lead_time(data_dir)
    cd = backtest_change_detection(data_dir)

    print("=== Lead time vs historical incidents ===")
    for r in lt["per_incident"]:
        if r["lead_time_weeks"] is not None:
            print(f"  {r['rack_id']:10s} {r['resource']:14s} event {r['event_date']} "
                  f"-> warned {r['lead_time_weeks']} weeks ahead")
        else:
            print(f"  {r['rack_id']:10s} {r['resource']:14s} event {r['event_date']} -> NOT caught in time")

    print("\n=== Regime-change detection / false alarms ===")
    print(f"  Regime-change recall: {cd['regime_change_recall']} ({cd['n_regime_changes']} injected)")
    print(f"  False-alarm rate on self-resolving spikes: {cd['false_alarm_rate_on_spikes']} "
          f"({cd['n_transient_spikes']} injected)")

    out = {"lead_time": lt, "change_detection": cd}
    with open(base / "reports" / "backtest_output.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
