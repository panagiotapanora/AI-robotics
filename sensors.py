#!/usr/bin/env python3
"""
sensors.py
Sensor primitives and simulator thread.
"""
import threading
import time
import random
import logging

logger = logging.getLogger(__name__)

class Sensor:
    def __init__(self, name, unit, value=0.0, drift=0.5, variance=1.0, fail_rate=0.0005):
        self.name = name
        self.unit = unit
        self.value = float(value)
        self.drift = float(drift)
        self.variance = float(variance)
        self.fail_rate = float(fail_rate)
        self.failed = False
        self.last_update = time.time()
        self.lock = threading.Lock()

    def step(self):
        """Advance sensor state by one update step (may fail)."""
        with self.lock:
            if self.failed:
                return self.value
            # Random failure simulation
            if random.random() < self.fail_rate:
                self.failed = True
                return self.value
            # normal update
            noise = random.gauss(0, self.variance)
            drift_delta = (random.random() - 0.5) * self.drift
            self.value = max(0.0, self.value + drift_delta + noise)
            self.last_update = time.time()
            return self.value

    def read(self):
        """Return current value or None if sensor failed."""
        with self.lock:
            if self.failed:
                return None
            return self.value

    def repair(self):
        with self.lock:
            self.failed = False
            self.last_update = time.time()

class SensorSimulator(threading.Thread):
    """
    Background thread that calls step() on every sensor at a fixed interval.
    """
    def __init__(self, sensors, update_interval=1.0):
        super().__init__(daemon=True)
        self.sensors = sensors # dict name -> Sensor
        self.update_interval = float(update_interval)
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            for s in list(self.sensors.values()):
                try:
                    s.step()
                except Exception as e:
                    # protect simulator thread from sensor exceptions
                    logger.error(f"Sensor step failed for {s.name}: {e}")
            time.sleep(self.update_interval)

    def stop(self):
        self._stop_event.set()
