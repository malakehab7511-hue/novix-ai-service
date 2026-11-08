"""Exact NOVIX raw-sensor feature reconstruction from the supplied equations.

Time lookup: last sample with timestamp <= target time.
Rolling windows: open on the left, closed on the right: (t-window, t].
Variance-named features intentionally use sample standard deviation (ddof=1).
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import math
import numpy as np

FEATURE_ORDER = [
    "f_mq2_ratio", "f_mq6_ratio",
    "f_mq2_slope_10s", "f_mq2_slope_30s", "f_mq2_accel", "f_mq2_var_10s",
    "f_mq6_slope_10s", "f_mq6_slope_30s", "f_mq6_accel", "f_mq6_var_10s",
    "f_mq_agreement", "f_temp_change_30s", "f_humidity_change_30s",
    "f_flame", "f_flame_frac_30s", "f_sensor_saturation",
]

@dataclass
class RawSensorFeatureExtractor:
    history_seconds: float = 45.0
    max_sample_gap_seconds: float = 3.0
    history: deque = field(default_factory=deque)
    last_time: Optional[float] = None

    def reset(self):
        self.history.clear()
        self.last_time = None

    @staticmethod
    def _at_or_before(records, target, key):
        found = None
        for row in records:
            if row["t"] <= target + 1e-6:
                found = row
            else:
                break
        return None if found is None else found[key]

    @staticmethod
    def _window(records, now, seconds, key):
        # Exactly (t-seconds, t]
        return [row[key] for row in records if now-seconds + 1e-6 < row["t"] <= now + 1e-6]

    def update(self, sample: Dict[str, Any], elapsed_seconds: float) -> Optional[Dict[str, float]]:
        required = ["mq2", "mq6", "temp", "humidity", "flame", "baseline_mq2", "baseline_mq6"]
        missing = [k for k in required if sample.get(k) is None]
        if missing:
            raise ValueError(f"Missing required raw inputs/baselines: {missing}")
        t = float(elapsed_seconds)
        if not math.isfinite(t):
            raise ValueError("elapsed_seconds must be finite")
        if self.last_time is not None and t <= self.last_time:
            raise ValueError("Samples must arrive in strictly increasing time order; call reset() at a new session")
        if self.last_time is not None and t-self.last_time > self.max_sample_gap_seconds:
            # A dropped/removed interval invalidates temporal windows; do not bridge stale history.
            self.reset()
        mq2, mq6 = float(sample["mq2"]), float(sample["mq6"])
        b2, b6 = float(sample["baseline_mq2"]), float(sample["baseline_mq6"])
        if b2 <= 0 or b6 <= 0:
            raise ValueError("MQ2/MQ6 baselines must be positive")
        flame = float(sample["flame"])
        if flame not in (0.0, 1.0):
            raise ValueError("flame must be a raw digital 0/1 value")

        row = {
            "t": t, "mq2": mq2 / b2, "mq6": mq6 / b6,
            "temp": float(sample["temp"]), "humidity": float(sample["humidity"]),
            "flame": flame,
        }
        self.history.append(row)
        self.last_time = t
        while self.history and t - self.history[0]["t"] > self.history_seconds:
            self.history.popleft()
        records = list(self.history)

        def at(delta, key):
            return self._at_or_before(records, t-delta, key)
        r2_10, r2_30 = at(10, "mq2"), at(30, "mq2")
        r6_10, r6_30 = at(10, "mq6"), at(30, "mq6")
        temp30, humidity30 = at(30, "temp"), at(30, "humidity")
        # Current 10-second slopes use fixed denominator 10, not observed dt.
        s2_10 = (row["mq2"] - r2_10) / 10.0 if r2_10 is not None else None
        s2_30 = (row["mq2"] - r2_30) / 30.0 if r2_30 is not None else None
        s6_10 = (row["mq6"] - r6_10) / 10.0 if r6_10 is not None else None
        s6_30 = (row["mq6"] - r6_30) / 30.0 if r6_30 is not None else None
        # Acceleration uses the previously computed 10s slope at the last sample <= t-10.
        prev_s2_10 = at(10, "s2_10")
        prev_s6_10 = at(10, "s6_10")
        row["s2_10"], row["s6_10"] = s2_10, s6_10

        w2 = self._window(records, t, 10, "mq2")
        w6 = self._window(records, t, 10, "mq6")
        wf = self._window(records, t, 30, "flame")
        if len(w2) < 2 or len(w6) < 2:
            return None
        if any(v is None for v in [r2_10, r2_30, r6_10, r6_30, temp30, humidity30,
                                    prev_s2_10, prev_s6_10]) or len(wf) == 0:
            return None
        agreement = row["mq2"] / row["mq6"] if row["mq6"] != 0 else 0.0
        features = {
            "f_mq2_ratio": row["mq2"],
            "f_mq6_ratio": row["mq6"],
            "f_mq2_slope_10s": s2_10,
            "f_mq2_slope_30s": s2_30,
            "f_mq2_accel": s2_10 - prev_s2_10,
            "f_mq2_var_10s": float(np.std(w2, ddof=1)),
            "f_mq6_slope_10s": s6_10,
            "f_mq6_slope_30s": s6_30,
            "f_mq6_accel": s6_10 - prev_s6_10,
            "f_mq6_var_10s": float(np.std(w6, ddof=1)),
            "f_mq_agreement": agreement,
            "f_temp_change_30s": row["temp"] - temp30,
            "f_humidity_change_30s": row["humidity"] - humidity30,
            "f_flame": flame,
            "f_flame_frac_30s": float(np.mean(wf)),
            "f_sensor_saturation": float(mq2 >= 4095 or mq6 >= 4095),
        }
        if list(features) != FEATURE_ORDER:
            raise RuntimeError("Feature order mismatch")
        if not all(np.isfinite(v) for v in features.values()):
            raise RuntimeError("Non-finite feature produced; refusing inference")
        return features
