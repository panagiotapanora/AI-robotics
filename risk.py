#!/usr/bin/env python3
"""
risk.py
A small explainable risk model using exponential smoothing and short-term variance.
No external dependencies.
"""
from collections import deque

class RiskModel:
    """
    Uses exponential smoothing per sensor to estimate trend and computes
    a combined risk score in range 0..100.
    """
    def __init__(self, sensors, alpha=0.3, history_len=20):
        # alpha: smoothing factor for exponential smoothing
        self.alpha = float(alpha)
        # smoothed values per sensor (name -> float or None)
        self.smoothed = {name: (s.read() if s.read() is not None else 0.0) for name, s in sensors.items()}
        # variance approximation per sensor
        self.variance = {name: 0.0 for name in sensors.keys()}
        # short history for each sensor to compute variance
        self.history = {name: deque(maxlen=history_len) for name in sensors.keys()}

    def update_sensors(self, sensors):
        """
        Update smoothed values and short-term variance from live sensors.
        sensors: dict name -> Sensor
        """
        for name, s in sensors.items():
            val = s.read()
            if val is None:
                # if sensor failed, do not update history or smoothing strongly; keep previous smoothed
                continue
            if name not in self.smoothed:
                self.smoothed[name] = val
            prev = self.smoothed[name]
            sm = self.alpha * val + (1 - self.alpha) * prev
            self.smoothed[name] = sm
            hist = self.history.setdefault(name, deque(maxlen=self.history[name].maxlen if name in self.history else 20))
            hist.append(val)
            if len(hist) >= 2:
                mean = sum(hist) / len(hist)
                var = sum((x - mean) ** 2 for x in hist) / len(hist)
            else:
                var = 0.0
            self.variance[name] = var

    def compute_score(self, rules):
        """
        Compute combined risk:
        - Sensors with thresholds contribute more.
        - Contribution is based on normalized exceedance or normalized smoothed value.
        - Volatility (variance) increases contribution.
        Returns dict {score: float, level: str}.
        """
        total = 0.0
        weight_total = 0.0
        for name, sm in self.smoothed.items():
            thr = rules.get_threshold(name)
            weight = 2.0 if thr is not None else 1.0
            weight_total += weight

            # if threshold exists, compute percent above threshold using smoothed value
            if thr is not None and thr > 0:
                pct = max(0.0, (sm - thr) / thr)
                contrib = min(1.0, pct * 2.0)
            else:
                # heuristic normalization for sensors without threshold
                contrib = min(1.0, sm / (sm + 50.0))

            vol = self.variance.get(name, 0.0)
            vol_factor = min(1.0, vol / (vol + 10.0))
            sensor_score = contrib * (0.6 + 0.4 * vol_factor)
            total += weight * sensor_score

        if weight_total <= 0:
            return {"score": 0.0, "level": "Low"}

        norm = (total / weight_total) * 100.0
        level = "Low"
        if norm > 75:
            level = "Critical"
        elif norm > 50:
            level = "High"
        elif norm > 25:
            level = "Medium"
        return {"score": round(norm, 2), "level": level}
