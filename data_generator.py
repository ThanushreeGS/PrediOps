"""
data_generator.py
------------------
Generates synthetic, multi-tenant data-centre telemetry matching the
hackathon problem statement (AO-02): 12 months of daily telemetry for
power, thermal, storage, network and VM-count, across several racks
and tenants, plus capacity limits/safety margins and a historical
incident log to validate forecasts against.

The generator deliberately injects THREE kinds of behaviour so the
rest of the pipeline has something real to detect:

  1. Steady linear/organic growth              (the "boring" baseline)
  2. Regime changes (step changes in slope)     -> tenant onboarding,
                                                    policy change
  3. Transient spikes that resolve on their own -> must NOT be mistaken
                                                    for a trend (false-
                                                    alarm test cases)

Tenant-level series are generated and stored separately so that
downstream code can enforce tenant isolation (never mixing tenant A's
raw rows into tenant B's view).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

RNG_SEED = 42
N_DAYS = 365
START_DATE = pd.Timestamp("2025-10-29")  # 12 months ending ~ "today"

RESOURCES = ["storage_pct", "power_kw", "thermal_c", "network_mbps", "vm_count"]

# storage/power/network/VMs are per-tenant and genuinely additive at the
# rack level (total rack storage used = sum of each tenant's usage).
# Ambient temperature is a shared PHYSICAL property of the rack, not
# additive across tenants -> modelled once per rack, attributed to
# "facility (shared load)" rather than a single tenant.
TENANT_ADDITIVE_RESOURCES = ["storage_pct", "power_kw", "network_mbps", "vm_count"]
RACK_SHARED_RESOURCES = ["thermal_c"]

# per-resource hard capacity limit and the safety-margin at which an
# operator wants to be warned (matches "capacity limits and safety
# margins for each resource" in the Resources section)
CAPACITY_LIMITS = {
    "storage_pct":   {"limit": 100.0, "safety_margin": 85.0, "unit": "%"},
    "power_kw":      {"limit": 480.0, "safety_margin": 420.0, "unit": "kW"},
    "thermal_c":     {"limit": 32.0,  "safety_margin": 28.0,  "unit": "°C"},
    "network_mbps":  {"limit": 10000.0, "safety_margin": 8500.0, "unit": "Mbps"},
    "vm_count":      {"limit": 400.0, "safety_margin": 360.0, "unit": "VMs"},
}


@dataclass
class RegimeChange:
    day: int
    resource: str
    slope_multiplier: float
    description: str


@dataclass
class TransientSpike:
    day: int
    resource: str
    magnitude: float
    duration_days: int
    description: str


@dataclass
class RackTenant:
    rack_id: str
    tenant_id: str
    base_values: dict
    daily_growth: dict
    noise_std: dict
    regime_changes: list = field(default_factory=list)
    spikes: list = field(default_factory=list)


def _build_topology():
    """Defines the synthetic facility: 4 racks, 6 tenants, with tenants
    sharing racks (so attribution has something to disentangle). Only
    the TENANT_ADDITIVE_RESOURCES are set here; thermal_c is defined
    separately per rack in `_build_rack_thermal` since it's a shared
    physical property, not a per-tenant quantity."""
    topology = [
        RackTenant(
            rack_id="rack-A1", tenant_id="tenant-nimbus",
            base_values={"storage_pct": 38, "power_kw": 180,
                         "network_mbps": 2200, "vm_count": 95},
            daily_growth={"storage_pct": 0.028, "power_kw": 0.06,
                          "network_mbps": 2.0, "vm_count": 0.03},
            noise_std={"storage_pct": 0.25, "power_kw": 1.8,
                       "network_mbps": 45, "vm_count": 0.6},
            regime_changes=[
                RegimeChange(210, "storage_pct", 4.5, "tenant-nimbus onboarded a new analytics workload"),
                RegimeChange(210, "vm_count", 3.0, "VM fleet scaled up alongside analytics workload"),
            ],
            spikes=[
                TransientSpike(95, "network_mbps", 3200, 4, "one-off batch export spike (self-resolved)"),
            ],
        ),
        RackTenant(
            rack_id="rack-A1", tenant_id="tenant-forge",
            base_values={"storage_pct": 22, "power_kw": 90,
                         "network_mbps": 900, "vm_count": 40},
            daily_growth={"storage_pct": 0.010, "power_kw": 0.02,
                          "network_mbps": 0.6, "vm_count": 0.01},
            noise_std={"storage_pct": 0.15, "power_kw": 1.0,
                       "network_mbps": 25, "vm_count": 0.4},
            spikes=[
                TransientSpike(150, "storage_pct", 6.0, 6, "temp log retention bug, rolled back (self-resolved)"),
            ],
        ),
        RackTenant(
            rack_id="rack-B2", tenant_id="tenant-orbital",
            base_values={"storage_pct": 55, "power_kw": 210,
                         "network_mbps": 3400, "vm_count": 140},
            daily_growth={"storage_pct": 0.035, "power_kw": 0.07,
                          "network_mbps": 3.2, "vm_count": 0.05},
            noise_std={"storage_pct": 0.3, "power_kw": 2.2,
                       "network_mbps": 60, "vm_count": 0.8},
            regime_changes=[
                RegimeChange(270, "power_kw", 3.2, "backup policy change: nightly full backups instead of incremental"),
            ],
        ),
        RackTenant(
            rack_id="rack-B2", tenant_id="tenant-quill",
            base_values={"storage_pct": 18, "power_kw": 70,
                         "network_mbps": 600, "vm_count": 28},
            daily_growth={"storage_pct": 0.006, "power_kw": 0.015,
                          "network_mbps": 0.4, "vm_count": 0.008},
            noise_std={"storage_pct": 0.12, "power_kw": 0.9,
                       "network_mbps": 20, "vm_count": 0.3},
        ),
        RackTenant(
            rack_id="rack-C3", tenant_id="tenant-helios",
            base_values={"storage_pct": 60, "power_kw": 260,
                         "network_mbps": 5200, "vm_count": 180},
            daily_growth={"storage_pct": 0.045, "power_kw": 0.09,
                          "network_mbps": 5.0, "vm_count": 0.07},
            noise_std={"storage_pct": 0.35, "power_kw": 2.5,
                       "network_mbps": 80, "vm_count": 1.0},
            regime_changes=[
                RegimeChange(305, "network_mbps", 5.5, "seasonal traffic ramp (Q4 promotional event)"),
            ],
            spikes=[
                TransientSpike(180, "vm_count", 15, 5, "short-lived load-test VMs, torn down after (self-resolved)"),
            ],
        ),
        RackTenant(
            rack_id="rack-C3", tenant_id="tenant-basalt",
            base_values={"storage_pct": 30, "power_kw": 110,
                         "network_mbps": 1500, "vm_count": 60},
            daily_growth={"storage_pct": 0.015, "power_kw": 0.03,
                          "network_mbps": 1.0, "vm_count": 0.02},
            noise_std={"storage_pct": 0.2, "power_kw": 1.3,
                       "network_mbps": 30, "vm_count": 0.5},
        ),
    ]
    return topology


def _build_rack_thermal():
    """Per-rack (not per-tenant) ambient temperature config, since
    temperature is a shared physical property driven by cumulative
    rack load rather than any single tenant's resource consumption."""
    return {
        "rack-A1": RackTenant(
            rack_id="rack-A1", tenant_id="facility-shared",
            base_values={"thermal_c": 22.5}, daily_growth={"thermal_c": 0.0015},
            noise_std={"thermal_c": 0.15},
        ),
        "rack-B2": RackTenant(
            rack_id="rack-B2", tenant_id="facility-shared",
            base_values={"thermal_c": 24.0}, daily_growth={"thermal_c": 0.002},
            noise_std={"thermal_c": 0.18},
            regime_changes=[
                RegimeChange(270, "thermal_c", 2.5, "higher sustained load after backup-policy change raised rack intake temperature"),
            ],
        ),
        "rack-C3": RackTenant(
            rack_id="rack-C3", tenant_id="facility-shared",
            base_values={"thermal_c": 25.5}, daily_growth={"thermal_c": 0.0028},
            noise_std={"thermal_c": 0.2},
            spikes=[
                TransientSpike(40, "thermal_c", 2.5, 3, "CRAC unit hiccup, auto-recovered (self-resolved)"),
            ],
        ),
    }


def generate_telemetry(out_dir: Path) -> dict:
    """Generates and writes:
       - telemetry.csv       (long format: date, rack_id, tenant_id, resource, value)
       - capacity_limits.csv (resource, limit, safety_margin, unit)
       - incident_log.csv    (historical capacity-related events, for backtesting)
       - ground_truth.csv    (the injected regime changes / spikes - for grading only,
                               NOT fed to the model)
    """
    rng = np.random.default_rng(RNG_SEED)
    topology = _build_topology() + list(_build_rack_thermal().values())
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    rows = []
    ground_truth_rows = []

    for rt in topology:
        resources_here = list(rt.base_values.keys())
        series = {r: np.zeros(N_DAYS) for r in resources_here}
        for r in resources_here:
            series[r][0] = rt.base_values[r]

        # build per-day effective growth rate (can change at regime points)
        growth_rate = {r: np.full(N_DAYS, rt.daily_growth[r]) for r in resources_here}
        for rc in rt.regime_changes:
            growth_rate[rc.resource][rc.day:] *= rc.slope_multiplier
            ground_truth_rows.append({
                "rack_id": rt.rack_id, "tenant_id": rt.tenant_id,
                "event_type": "regime_change", "resource": rc.resource,
                "day_index": rc.day, "date": dates[rc.day].date().isoformat(),
                "description": rc.description,
            })

        for r in resources_here:
            for t in range(1, N_DAYS):
                series[r][t] = series[r][t - 1] + growth_rate[r][t] + rng.normal(0, rt.noise_std[r])

        # overlay transient spikes (decay back to trend within duration_days)
        for sp in rt.spikes:
            end = min(sp.day + sp.duration_days, N_DAYS)
            decay = np.linspace(sp.magnitude, 0, end - sp.day)
            series[sp.resource][sp.day:end] += decay
            ground_truth_rows.append({
                "rack_id": rt.rack_id, "tenant_id": rt.tenant_id,
                "event_type": "transient_spike", "resource": sp.resource,
                "day_index": sp.day, "date": dates[sp.day].date().isoformat(),
                "description": sp.description,
            })

        # clip physically-sane series
        if "thermal_c" in series:
            series["thermal_c"] = np.clip(series["thermal_c"], 15, None)
        if "vm_count" in series:
            series["vm_count"] = np.clip(series["vm_count"], 0, None)
        if "storage_pct" in series:
            series["storage_pct"] = np.clip(series["storage_pct"], 0, None)

        for r in resources_here:
            for t in range(N_DAYS):
                rows.append({
                    "date": dates[t], "rack_id": rt.rack_id, "tenant_id": rt.tenant_id,
                    "resource": r, "value": round(float(series[r][t]), 4),
                })

    telemetry = pd.DataFrame(rows)
    limits = pd.DataFrame([
        {"resource": r, **v} for r, v in CAPACITY_LIMITS.items()
    ])
    ground_truth = pd.DataFrame(ground_truth_rows)

    # Build an incident log: historical dates where a RACK-level aggregate
    # resource actually crossed its safety margin -> these are the events
    # a good forecaster should have warned about in advance.
    def _agg_func(resource):
        return "mean" if resource in RACK_SHARED_RESOURCES else "sum"

    agg_parts = []
    for resource, part in telemetry.groupby("resource"):
        agg_parts.append(
            part.groupby(["date", "rack_id", "resource"])["value"]
            .agg(_agg_func(resource)).reset_index()
        )
    agg = pd.concat(agg_parts, ignore_index=True)
    incident_rows = []
    for (rack, resource), grp in agg.groupby(["rack_id", "resource"]):
        grp = grp.sort_values("date")
        margin = CAPACITY_LIMITS[resource]["safety_margin"]
        crossed = grp[grp["value"] >= margin]
        if not crossed.empty:
            first_cross = crossed.iloc[0]
            incident_rows.append({
                "rack_id": rack, "resource": resource,
                "event_date": first_cross["date"].date().isoformat(),
                "value_at_event": round(float(first_cross["value"]), 2),
                "safety_margin": margin,
                "description": f"{resource} on {rack} crossed safety margin",
            })
    incident_log = pd.DataFrame(incident_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(out_dir / "telemetry.csv", index=False)
    limits.to_csv(out_dir / "capacity_limits.csv", index=False)
    incident_log.to_csv(out_dir / "incident_log.csv", index=False)
    ground_truth.to_csv(out_dir / "ground_truth.csv", index=False)

    return {
        "telemetry": telemetry, "limits": limits,
        "incident_log": incident_log, "ground_truth": ground_truth,
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    data = generate_telemetry(base / "data")
    print("telemetry rows:", len(data["telemetry"]))
    print("incidents:", len(data["incident_log"]))
    print("injected ground-truth events:", len(data["ground_truth"]))
    print("Files successfully generated in:", base / "data")
