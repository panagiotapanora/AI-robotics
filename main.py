#!/usr/bin/env python3
"""
main.py
Project entrypoint. Builds sensors, rule engine and risk model and runs interactive console.
Merged with 2D robot simulator integration (sensors <- simulator, rule safety override, unified system).
"""

import time
import logging
from collections import deque
from sensors import Sensor, SensorSimulator
from rules import RuleEngine
from risk import RiskModel
import app
import threading
import json
from ml_model import MLModel

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# simulator imports
from simulator import (
    Environment, Obstacle, Robot, Simulator, SimulatorConfig,
    simple_wall_follow_controller_factory
)

def print_menu(safety_override_enabled, sim_running):
    print("=" * 60)
    print("Robotic-AI Environmental Monitor (Console)".center(60))
    print("=" * 60)
    print("1) Dashboard (live) - d, dashboard")
    print("2) Set Threshold - t, threshold")
    print("3) Add Custom Sensor - a, add")
    print("4) Show Status (one-shot) - s, show")
    print("5) Export Logs (CSV copy) - e, export")
    print("6) Save Config - v, save")
    print("7) Snapshot (save txt) - p, snapshot")
    print("8) Stop & Exit - x, stop")
    print("-" * 60)
    print("Robot simulator options:")
    print(" r) Toggle Robot Simulator (start/stop) - r")
    print(f" o) Toggle Safety Override (RuleEngine used as safety): {'ON' if safety_override_enabled else 'OFF'} - o")
    print(" u) Unified demo (sim + ML controller) - u")
    print("=" * 60)

def set_threshold_interactive(sensors, rules):
    print("Current sensors:")
    for name in sensors.keys():
        print(" -", name)
    target = input("Sensor name to set threshold for: ").strip()

    if not target:
        print("Sensor name cannot be empty.")
        return

    if target not in sensors:
        print(f"Sensor '{target}' not found.")
        return

    try:
        val_str = input(f"Enter threshold value for {target}: ").strip()
        if not val_str:
            print("Threshold value cannot be empty.")
            return

        val = float(val_str)

        if val < 0:
            print("Warning: Setting negative threshold.")

        rules.set_threshold(target, val)
        print(f"Threshold set: {target} -> {val}")
    except ValueError as e:
        logger.warning(f"Invalid threshold input: {e}")
        print("Invalid value. Please enter a number.")

def add_sensor_interactive(sensors, rules, risk, sensors_lock=None):
    name = input("New sensor name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    # Validate sensor name (no special characters that could cause issues)
    if any(char in name for char in ['/', '\\', '\0', '\n', '\r']):
        print("Sensor name contains invalid characters.")
        return

    # Check for existing sensor (thread-safe read)
    if sensors_lock:
        with sensors_lock:
            if name in sensors:
                print(f"Sensor '{name}' already exists.")
                return
    else:
        if name in sensors:
            print(f"Sensor '{name}' already exists.")
            return

    unit = input("Unit (e.g., °C, %, ppm): ").strip() or "u"

    try:
        init_val_str = input("Initial value (number): ").strip()
        if init_val_str:
            init_val = float(init_val_str)
        else:
            print("Using default initial value: 0.0")
            init_val = 0.0
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid initial value input: {e}. Using 0.0")
        print("Invalid initial value, using 0.0")
        init_val = 0.0

    try:
        drift_str = input("Drift per step (suggest 0.1-2.0): ").strip()
        if drift_str:
            drift = float(drift_str)
            if drift < 0:
                print("Warning: Negative drift will tend to decrease sensor values.")
        else:
            drift = 0.5
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid drift input: {e}. Using 0.5")
        drift = 0.5

    try:
        var_str = input("Noise variance (suggest 0.5-10.0): ").strip()
        if var_str:
            var = float(var_str)
            if var < 0:
                print("Warning: Negative variance is unusual. Using absolute value.")
                var = abs(var)
        else:
            var = 1.0
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid variance input: {e}. Using 1.0")
        var = 1.0

    # Add sensor with thread-safe lock
    if sensors_lock:
        with sensors_lock:
            sensors[name] = Sensor(name, unit, value=init_val, drift=drift, variance=var)
            # update risk model internals
            risk.smoothed[name] = init_val
            risk.variance[name] = 0.0
            risk.history[name] = deque(maxlen=20)
    else:
        sensors[name] = Sensor(name, unit, value=init_val, drift=drift, variance=var)
        # update risk model internals
        risk.smoothed[name] = init_val
        risk.variance[name] = 0.0
        risk.history[name] = deque(maxlen=20)

    print(f"Sensor '{name}' added successfully.")

def show_status(sensors, rules, risk, sim=None):
    app.clear_console()
    risk.update_sensors(sensors)
    sc = risk.compute_score(rules)
    print("Current Readings:")
    for name, s in sensors.items():
        val = s.read()
        thr = rules.get_threshold(name)
        thr_str = "-" if thr is None else thr
        val_str = val if val is not None else "FAILED"
        print(f" - {name:15} {val_str:9} {s.unit:3} Threshold: {thr_str}")
    print(f"\nAI Risk: {sc['score']} Level: {sc['level']}")
    print("\nRecent alerts:")
    for a in rules.alerts[-10:]:
        print(f" - {a['timestamp']} | {a['sensor']} = {a['value']} > {a['threshold']}")
    if sim is not None:
        print("\nRobot pose:", sim.robot_pose())
        print("Sim time:", sim.time)

def main_menu():
    # --- original sensors
    sensors = {
        "Temperature": Sensor("Temperature", "°C", value=22.0, drift=0.2, variance=0.8),
        "Humidity": Sensor("Humidity", "%", value=45.0, drift=0.3, variance=2.0),
        "CO2": Sensor("CO2", "ppm", value=420.0, drift=1.0, variance=8.0),
    }
    # Thread lock for sensor dictionary modifications
    sensors_lock = threading.Lock()

    rules = RuleEngine()
    # default thresholds
    rules.set_threshold("Temperature", 35.0)
    rules.set_threshold("CO2", 1000.0)
    app.load_config(rules)
    app.init_logs()
    risk = RiskModel(sensors)
    sim_sensor_sim = SensorSimulator(sensors, update_interval=1.0)
    sim_sensor_sim.start()

    # --- add robot-specific sensors into the same sensors dict so everything is unified
    # min_distance in meters and bump as binary sensor
    with sensors_lock:
        sensors["min_distance"] = Sensor("min_distance", "m", value=3.0, drift=0.0, variance=0.01)
        sensors["bump"] = Sensor("bump", "", value=0.0, drift=0.0, variance=0.0)

    # Robot simulator components (not running by default)
    env = Environment(width=6.0, height=6.0, obstacles=[Obstacle(3.0, 3.0, 0.6), Obstacle(1.2, 4.0, 0.3)])
    robot = Robot(x=0.7, y=0.7, theta=0.0)
    sim_cfg = SimulatorConfig()
    sim = Simulator(env, robot, sim_cfg)
    # default controller (simple wall follow)
    base_controller = simple_wall_follow_controller_factory(0.5)
    sim.controller = base_controller

    # control flags
    sim_thread = None
    sim_thread_stop = [False]
    sim_running = False
    safety_override_enabled = False
    ml_model = MLModel()

    # logger thread flag
    stop_logger_flag = [False]

    # background logger thread (keeps existing logging behavior)
    def logger_loop():
        while not stop_logger_flag[0]:
            for name, s in list(sensors.items()):
                val = s.read()
                app.log_reading(name, val)
                ev = rules.evaluate(name, val)
                if ev:
                    app.log_alert(ev)
            time.sleep(2.0)

    lg_thread = threading.Thread(target=logger_loop, daemon=True)
    lg_thread.start()

    # robot runner loop (runs simulator stepping and updates sensors)
    def robot_runner():
        # Store original controller and create a wrapper that enforces RuleEngine as safety override
        original_controller = sim.controller

        def controller_wrapper(obs):
            # first check safety override
            min_dist = min(obs.distances) if obs.distances else float('inf')
            if safety_override_enabled and rules.get_threshold("min_distance") is not None:
                thr = rules.get_threshold("min_distance")
                # If threshold triggers (value > thr in RuleEngine semantics),
                # for min_distance we actually want to check min_dist < thr -> emergency stop.
                # To reuse RuleEngine, we use a custom check here.
                try:
                    t = float(thr)
                    if min_dist < t:
                        # emergency stop
                        return 0.0, 0.0
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid threshold value for min_distance: {e}")
            # otherwise call the original controller
            try:
                return original_controller(obs)
            except Exception as e:
                logger.error(f"Controller execution failed: {e}")
                return 0.0, 0.0

        # Replace sim.controller with wrapper
        sim.controller = controller_wrapper

        # run loop
        while not sim_thread_stop[0]:
            obs = sim.step()
            # update unified sensors with robot data
            try:
                sensors["min_distance"].value = min(obs.distances) if obs.distances else None
            except Exception as e:
                logger.error(f"Failed to update min_distance sensor: {e}")
                sensors["min_distance"].value = None
            sensors["bump"].value = 1.0 if obs.bump else 0.0

            # evaluate robot sensors with rules (we treat min_distance specially: threshold is distance in meters)
            # If threshold is set and min_dist < threshold -> create a special alert
            thr = rules.get_threshold("min_distance")
            if thr is not None and sensors["min_distance"].value is not None:
                try:
                    if sensors["min_distance"].value < float(thr):
                        # produce an alert like other sensors (timestamped)
                        event = {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "sensor": "min_distance",
                            "value": sensors["min_distance"].value,
                            "threshold": float(thr)
                        }
                        rules.alerts.append(event)
                        app.log_alert(event)
                except Exception as e:
                    logger.error(f"Failed to log min_distance alert: {e}")

            # call optional step callback
            if sim.step_callback:
                try:
                    sim.step_callback(sim)
                except Exception as e:
                    logger.error(f"Step callback failed: {e}")

            # small sleep to match sim dt (or shorter)
            time.sleep(max(0.001, sim.cfg.dt))

    # function to start sim thread
    def start_sim():
        nonlocal sim_thread, sim_running, sim_thread_stop
        if sim_running:
            print("Simulator already running.")
            return
        sim_thread_stop[0] = False
        sim_thread = threading.Thread(target=robot_runner, daemon=True)
        sim_thread.start()
        sim_running = True
        print("Simulator started.")

    def stop_sim():
        nonlocal sim_thread, sim_running, sim_thread_stop
        if not sim_running:
            print("Simulator not running.")
            return
        sim_thread_stop[0] = True
        sim_running = False
        print("Simulator stopping...")

    try:
        while True:
            print_menu(safety_override_enabled, sim_running)
            cmd = input("Enter command: ").strip().lower()
            if cmd in ("1", "dashboard", "d"):
                # show live dashboard; pass robot pose / sim.time if sim running
                pose = sim.robot_pose() if sim_running else None
                sim_time = sim.time if sim_running else None
                app.dashboard_loop(sensors, rules, risk, refresh=1.0, robot_pose=pose, sim_time=sim_time)
                app.clear_console()
            elif cmd in ("2", "threshold", "t"):
                set_threshold_interactive(sensors, rules)
            elif cmd in ("3", "add", "a"):
                add_sensor_interactive(sensors, rules, risk, sensors_lock)
            elif cmd in ("4", "show", "s"):
                show_status(sensors, rules, risk, sim if sim_running else None)
            elif cmd in ("5", "export", "e"):
                out1, out2 = app.export_logs_simple()
                if out1 and out2:
                    print(f"Exported readings to {out1}")
                    print(f"Exported alerts to {out2}")
                else:
                    print("Export failed.")
            elif cmd in ("6", "save", "v"):
                app.save_config(rules)
                print("Config saved.")
            elif cmd in ("7", "snapshot", "p"):
                fname = f"snapshot_{int(time.time())}.txt"
                ok = app.dashboard_snapshot_text(sensors, rules, risk, fname)
                if ok:
                    print(f"Snapshot saved to {fname}")
                else:
                    print("Snapshot failed.")
            elif cmd in ("8", "stop", "x", "quit", "exit"):
                print("Stopping simulation and exiting...")
                break
            elif cmd == "r":
                # toggle sim running
                if sim_running:
                    stop_sim()
                else:
                    start_sim()
            elif cmd == "o":
                safety_override_enabled = not safety_override_enabled
                print("Safety override is now", "ENABLED" if safety_override_enabled else "DISABLED")
            elif cmd == "u":
                # unified demo: use MLModel as controller (very simple hook) if available
                print("Starting unified demo: ML controller (fallback to base controller if no model).")
                def ml_controller(obs):
                    # collect sensor_values for ml_model (Temperature, Humidity, CO2)
                    sensor_vals = {
                        "Temperature": sensors.get("Temperature").read() or 0.0,
                        "Humidity": sensors.get("Humidity").read() or 0.0,
                        "CO2": sensors.get("CO2").read() or 0.0
                    }
                    pred = ml_model.predict(sensor_vals)
                    # use probability to modulate speed
                    prob = pred.get("probability", 0.0)
                    # high probability -> slow down
                    v = max(0.0, 0.6 * (1.0 - prob))
                    # steering: fallback to base controller for angular
                    _, omega = base_controller(obs)
                    return v, omega
                sim.controller = ml_controller
                if not sim_running:
                    start_sim()
                print("Unified demo running (press 'r' to stop sim).")
            else:
                print("Unknown command. Choose number or keyword from menu.")
            print("\n")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        # stop background work
        stop_logger_flag[0] = True
        # stop simulator
        if sim_running:
            stop_sim()
            # give a moment
            time.sleep(0.1)
        # stop sensor simulator thread
        sim_sensor_sim.stop()
        sim_sensor_sim.join(timeout=1.0)
        # join logger thread
        lg_thread.join(timeout=1.0)

if __name__ == "__main__":
    main_menu()
