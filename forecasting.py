import numpy as np
import pandas as pd
from scipy.stats import theilslopes

N_BOOTSTRAP = 500
CONFIDENCE = 0.80  # 80% interval: honest, not falsely precise


def _current_regime_window(dates: pd.Series, values: np.ndarray, change_points: list,
                            min_window_days: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Slices the series to start at the most recent confirmed regime
    change, falling back to the last `min_window_days` if the regime
    is too young to fit reliably, or all data if there's no change."""
    n = len(values)
    if change_points:
        last_cp = max(change_points, key=lambda c: c["day_index"])
        start = last_cp["day_index"]
        if n - start < min_window_days:
            start = max(0, n - min_window_days)
    else:
        start = max(0, n - 180)  # default: last ~6 months if no regime detected
    idx = np.arange(start, n)
    return idx, values[start:n]


def forecast_time_to_exhaustion(dates: pd.Series, values: np.ndarray,
                                 limit: float, safety_margin: float,
                                 change_points: list) -> dict:
    n = len(values)
    idx, window_vals = _current_regime_window(dates, values, change_points)
    x = np.arange(len(window_vals), dtype=float)

    if len(window_vals) < 8:
        return {"status": "insufficient_data"}

    slope, intercept, slope_lo, slope_hi = theilslopes(window_vals, x, alpha=1 - CONFIDENCE)
    fitted = intercept + slope * x
    residuals = window_vals - fitted
    today_value = float(window_vals[-1])
    today_idx_global = int(idx[-1])
    today_date = pd.Timestamp(dates.iloc[today_idx_global])

    def days_to_cross(target: float, slope_: float) -> float | None:
        if slope_ <= 1e-6:
            return None
        remaining = target - today_value
        if remaining <= 0:
            return 0.0
        return remaining / slope_

    point_days_margin = days_to_cross(safety_margin, slope)
    point_days_limit = days_to_cross(limit, slope)

    # bootstrap residuals -> distribution of days-to-cross for an honest range
    rng = np.random.default_rng(7)
    boot_days_margin, boot_days_limit = [], []
    if slope > 1e-6 and len(residuals) >= 8:
        for _ in range(N_BOOTSTRAP):
            resampled = rng.choice(residuals, size=len(residuals), replace=True)
            noisy_last = fitted[-1] + resampled[-1]
            jitter_slope = slope + rng.normal(0, max(1e-6, (slope_hi - slope_lo) / 4))
            dm = None if jitter_slope <= 1e-6 else max(0.0, (safety_margin - noisy_last) / jitter_slope)
            dl = None if jitter_slope <= 1e-6 else max(0.0, (limit - noisy_last) / jitter_slope)
            if dm is not None:
                boot_days_margin.append(dm)
            if dl is not None:
                boot_days_limit.append(dl)

    def summarize(point, boot):
        if point is None:
            return {"eta_days": None, "eta_date": None, "range_days": None, "range_dates": None}
        lo = float(np.percentile(boot, 10)) if boot else point
        hi = float(np.percentile(boot, 90)) if boot else point
        return {
            "eta_days": round(point, 1),
            "eta_date": str((today_date + pd.Timedelta(days=point)).date()),
            "range_days": [round(lo, 1), round(hi, 1)],
            "range_dates": [
                str((today_date + pd.Timedelta(days=lo)).date()),
                str((today_date + pd.Timedelta(days=hi)).date()),
            ],
        }

    return {
        "status": "ok",
        "as_of_date": str(today_date.date()),
        "current_value": round(today_value, 2),
        "daily_growth_rate": round(float(slope), 4),
        "growth_rate_range": [round(float(slope_lo), 4), round(float(slope_hi), 4)],
        "regime_window_days": len(window_vals),
        "regime_start_date": str(pd.Timestamp(dates.iloc[int(idx[0])]).date()),
        "to_safety_margin": summarize(point_days_margin, boot_days_margin),
        "to_hard_limit": summarize(point_days_limit, boot_days_limit),
    }
