"""
change_detection.py
--------------------
Detects "regime changes" (goal #2): points where the underlying GROWTH
RATE of a resource shifts persistently, as opposed to single-day noise
or a transient spike that reverts on its own.

Approach: sliding-window slope comparison. At each candidate day t, fit
a robust (Theil-Sen) trend to the W days immediately BEFORE t and the W
days immediately AFTER t, and compare the two slopes. Because each side
is a robust fit over a whole window (not a single day), a short-lived
spike (a few days, by construction shorter than W) barely moves either
window's slope and is naturally ignored -- this is what keeps the
false-alarm rate low without a separate, fragile "persistence" rule. A
genuine regime change (new tenant, policy change) shifts the AFTER
slope for the rest of the window and shows up as a large, sustained
slope delta.

Candidates are scored by |slope_after - slope_before|, non-maximum
suppression keeps only local peaks at least `min_separation_days`
apart, and a threshold (in units of the robust spread of slope deltas
across the whole series) decides what counts as "real".
"""

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=1, center=True).median().to_numpy()


def detect_change_points(dates: pd.Series, values: np.ndarray,
                          window_days: int = 30,
                          step_days: int = 5,
                          threshold_k: float = 1.9,
                          min_separation_days: int = 30) -> list[dict]:
    """
    Returns detected regime changes, sorted by day_index:
        [{"day_index": ..., "date": ..., "old_slope": ..., "new_slope": ...,
          "confidence": ...}, ...]
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < window_days * 2 + 10:
        return []

    smooth_vals = _smooth(values)

    candidates = list(range(window_days, n - window_days, step_days))
    if not candidates:
        return []

    deltas = []
    slopes_before = []
    slopes_after = []
    for t in candidates:
        before = smooth_vals[t - window_days:t]
        after = smooth_vals[t:t + window_days]
        sb, *_ = theilslopes(before, np.arange(len(before)))
        sa, *_ = theilslopes(after, np.arange(len(after)))
        slopes_before.append(sb)
        slopes_after.append(sa)
        deltas.append(sa - sb)

    deltas = np.array(deltas)
    # robust spread (MAD-based) of slope deltas, used as the noise scale
    mad = np.median(np.abs(deltas - np.median(deltas))) * 1.4826 + 1e-9
    threshold = threshold_k * mad

    # non-maximum suppression over |delta|, keeping points >= threshold
    order = np.argsort(-np.abs(deltas))
    kept_idx = []
    for i in order:
        if np.abs(deltas[i]) < threshold:
            break
        t = candidates[i]
        if all(abs(t - candidates[j]) >= min_separation_days for j in kept_idx):
            kept_idx.append(i)

    kept_idx.sort(key=lambda i: candidates[i])

    results = []
    for i in kept_idx:
        t = candidates[i]
        confidence = float(min(0.99, abs(deltas[i]) / (threshold + 1e-9) / 3))
        results.append({
            "day_index": int(t),
            "date": str(pd.Timestamp(dates.iloc[t]).date()),
            "old_slope": round(float(slopes_before[i]), 4),
            "new_slope": round(float(slopes_after[i]), 4),
            "confidence": round(confidence, 2),
        })
    return results
