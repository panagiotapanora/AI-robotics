#!/usr/bin/env python3
"""
app.py
Console UI utilities, persistence (config/logging), dashboard and interaction helpers.
Depends on sensors.Sensor, rules.RuleEngine, risk.RiskModel provided by other modules.
"""
import os
import csv
import json
import time
import logging
from datetime import datetime
from ml_model import MLModel
# keyboard detection: cross-platform
import sys

logger = logging.getLogger(__name__)
try:
    import msvcrt # Windows
    def kb_hit():
        return msvcrt.kbhit()
    def get_char():
        return msvcrt.getwch()
except Exception:
    import select, tty, termios
    def kb_hit():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(dr)
    def get_char():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

LOG_CSV = "sensor_log.csv"
ALERT_CSV = "alerts.csv"
CONFIG_JSON = "config.json"

def init_logs():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "sensor", "value"])
    if not os.path.exists(ALERT_CSV):
        with open(ALERT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "sensor", "value", "threshold"])

def log_reading(sensor_name, value):
    try:
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([datetime.utcnow().isoformat() + "Z", sensor_name, float(value) if value is not None else "NULL"])
    except Exception as e:
        logger.error(f"Failed to log reading for {sensor_name}: {e}")

def log_alert(event):
    try:
        with open(ALERT_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([event["timestamp"], event["sensor"], event["value"], event["threshold"]])
    except Exception as e:
        logger.error(f"Failed to log alert for {event.get('sensor', 'unknown')}: {e}")

def save_config(rules):
    data = {"thresholds": rules.thresholds}
    try:
        with open(CONFIG_JSON, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def load_config(rules):
    if not os.path.exists(CONFIG_JSON):
        return
    try:
        with open(CONFIG_JSON, "r") as f:
            data = json.load(f)
        for k, v in data.get("thresholds", {}).items():
            rules.set_threshold(k, v)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def human_time(ts=None):
    if ts is None:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return ts

def dashboard_snapshot_text(sensors, rules, risk, filename):
    """Write one-shot snapshot of dashboard to a text file (for screenshots in report)."""
    try:
        with open(filename, "w") as f:
            f.write(f"Snapshot UTC: {human_time()}\n")
            f.write("-" * 60 + "\n")
            for name, s in sensors.items():
                val = s.read()
                thr = rules.get_threshold(name)
                thr_str = f"{thr}" if thr is not None else "-"
                f.write(f"{name:12} {val if val is not None else 'FAILED':9} {s.unit:4} Threshold: {thr_str}\n")
            f.write("-" * 60 + "\n")
            score = risk.compute_score(rules)
            f.write(f"AI Risk Score: {score['score']} Level: {score['level']}\n")
            f.write("\nRecent alerts:\n")
            for a in rules.alerts[-10:]:
                f.write(f" - {a['timestamp']} | {a['sensor']} = {a['value']} > {a['threshold']}\n")
        return True
    except Exception as e:
        logger.error(f"Failed to create snapshot {filename}: {e}")
        return False

def export_logs_simple():
    """Copy CSVs to timestamped export files."""
    out_log = f"export_readings_{int(time.time())}.csv"
    out_alerts = f"export_alerts_{int(time.time())}.csv"
    try:
        if os.path.exists(LOG_CSV):
            with open(LOG_CSV, "r", newline="") as src, open(out_log, "w", newline="") as dst:
                dst.write(src.read())
        if os.path.exists(ALERT_CSV):
            with open(ALERT_CSV, "r", newline="") as src, open(out_alerts, "w", newline="") as dst:
                dst.write(src.read())
        return out_log, out_alerts
    except Exception as e:
        logger.error(f"Failed to export logs: {e}")
        return None, None

def dashboard_loop(sensors, rules, risk, refresh=1.0, robot_pose=None, sim_time=None):
    """
    Live dashboard: runs until user presses 'm'.
    Displays sensor values, AI risk, robot pose, sim time.
    """
    try:
        clear_console()
        print("Press 'enter' to stop/continue the refresh\n")
        print("Live Dashboard — Press 'm' to return to the menu (while the table is not refreshed)\n")
        while True:
            risk.update_sensors(sensors)
            score = risk.compute_score(rules)
            print(f"UTC Time: {human_time()}")
            print("-" * 60)
            
            # Sensors
            for name, s in sensors.items():
                val = s.read()
                thr = rules.get_threshold(name)
                thr_str = f"{thr}" if thr is not None else "-"
                alert_marker = "!" if (thr is not None and val is not None and val > thr) else " "
                print(f"{alert_marker} {name:12} {val if val is not None else 'FAILED':9} {s.unit:4}   Threshold: {thr_str}")

            print("-" * 60)
            
            # AI Risk
            print(f"AI Risk Score: {score['score']}   Level: {score['level']}")

            # Robot info
            if robot_pose:
                x, y, theta = robot_pose
                print(f"\nRobot Pose: x={x:.2f}, y={y:.2f}, θ={theta:.2f} rad")
            if sim_time is not None:
                print(f"Simulation Time: {sim_time:.2f} s")

            print("\nPress 'enter' to stop/continue the refresh")
            print("\nPress 'm' to return to the menu (while the table is not refreshed)")
            
            # short delay with control for 'm'
            wait_sec = 0.0
            while wait_sec < refresh:
                if kb_hit():
                    try:
                        ch = get_char()
                        if ch.lower() == 'm':
                            return  # return to menu

                    except Exception as e:
                        logger.debug(f"Keyboard input error: {e}")
                time.sleep(0.1)
                wait_sec += 0.1
            clear_console()
            
            print("Live Dashboard\n")

    except KeyboardInterrupt:
        return

