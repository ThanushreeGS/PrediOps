"""
attribution.py
---------------
Goal #3: "Point to the tenants or workloads driving each trend."

For a rack-level resource trend, decomposes the aggregate growth slope
into each tenant's individual contribution over the SAME regime window
used for the forecast, so the attribution and the forecast are always
talking about the same time period.

Tenant isolation note (Safety/Security requirement): this function is
the only place that touches more than one tenant's raw series at once,
and it is only ever called server-side to compute an AGGREGATED,
rack-level attribution summary (tenant name + % contribution). The
per-tenant raw telemetry itself is never joined into another tenant's
report - see report.py, which renders one tenant's own series only in
that tenant's drill-down view.
"""

import numpy as np
from scipy.stats import theilslopes


def attribute_rack_trend(tenant_series: dict[str, np.ndarray], window_start: int) -> list[dict]:
    """
    tenant_series: {tenant_id: full_value_array} for every tenant on the rack,
                   already aligned to the same date index as the rack aggregate.
    window_start:  index into those arrays where the current regime begins
                   (matches forecasting._current_regime_window's `idx[0]`).

    Returns tenants sorted by contribution, descending:
        [{"tenant_id": ..., "slope_per_day": ..., "share_of_rack_growth": ...}, ...]
    """
    contributions = []
    total_slope = 0.0
    per_tenant_slope = {}

    for tenant_id, series in tenant_series.items():
        window = series[window_start:]
        if len(window) < 8:
            continue
        x = np.arange(len(window), dtype=float)
        slope, *_ = theilslopes(window, x)
        slope = max(0.0, float(slope))  # only growth counts toward "driving the trend"
        per_tenant_slope[tenant_id] = slope
        total_slope += slope

    if total_slope <= 1e-9:
        return []

    for tenant_id, slope in per_tenant_slope.items():
        share = slope / total_slope
        contributions.append({
            "tenant_id": tenant_id,
            "slope_per_day": round(slope, 4),
            "share_of_rack_growth": round(share, 3),
        })

    contributions.sort(key=lambda c: c["share_of_rack_growth"], reverse=True)
    return contributions
