"""
audit.py
--------
Safety/Security requirement: "Keep an audit trail of forecasts and the
data they were based on."

Writes one append-only JSON-lines record per forecast run: what was
forecast, when, which data window it used, and a hash of the input
values so a later reviewer can verify the forecast wasn't silently
recomputed on different data. Never overwrites prior entries.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _hash_values(values: np.ndarray) -> str:
    return hashlib.sha256(np.round(values, 4).tobytes()).hexdigest()[:16]


def log_forecast(audit_dir: Path, rack_id: str, resource: str,
                  data_window: dict, forecast_result: dict,
                  change_points: list, attribution: list) -> dict:
    audit_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rack_id": rack_id,
        "resource": resource,
        "input_data_window": {
            "start_date": data_window["start_date"],
            "end_date": data_window["end_date"],
            "n_points": data_window["n_points"],
            "data_hash": data_window["data_hash"],
        },
        "detected_change_points": change_points,
        "forecast": forecast_result,
        "attribution": attribution,
    }
    log_path = audit_dir / "forecast_audit_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
