#!/usr/bin/env python3
"""
rules.py
Threshold-based rule engine and alert storage.
"""
from datetime import datetime

class RuleEngine:
    def __init__(self):
        # thresholds: sensor_name -> value
        self.thresholds = {}
        # alerts: list of dicts with timestamp, sensor, value, threshold
        self.alerts = []

    def set_threshold(self, sensor_name, value):
        self.thresholds[str(sensor_name)] = float(value)

    def get_threshold(self, sensor_name):
        return self.thresholds.get(sensor_name, None)

    def evaluate(self, sensor_name, value):
        """
        Evaluate a single sensor reading against its threshold.
        If value is None (sensor failure), do not raise threshold alert here.
        Returns the alert dict if threshold crossed, else False.
        """
        thr = self.get_threshold(sensor_name)
        if thr is None:
            return False
        if value is None:
            return False
        try:
            v = float(value)
        except Exception:
            return False
        if v > thr:
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sensor": sensor_name,
                "value": v,
                "threshold": float(thr)
            }
            self.alerts.append(event)
            return event
        return False
