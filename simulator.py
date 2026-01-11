# simulator.py
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict

Vector = Tuple[float, float]

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

@dataclass
class Obstacle:
    x: float
    y: float
    r: float  # radius (circular obstacle)

    def contains(self, px: float, py: float) -> bool:
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.r ** 2

@dataclass
class Environment:
    width: float = 10.0
    height: float = 10.0
    obstacles: List[Obstacle] = field(default_factory=list)

    def is_free(self, x: float, y: float) -> bool:
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        for o in self.obstacles:
            if o.contains(x, y):
                return False
        return True

@dataclass
class SensorReading:
    distances: List[float]
    bump: bool

@dataclass
class Robot:
    x: float
    y: float
    theta: float  # orientation radians (0 = +x)
    radius: float = 0.2
    v: float = 0.0
    omega: float = 0.0
    max_speed: float = 1.0
    max_omega: float = math.pi

@dataclass
class SimulatorConfig:
    dt: float = 0.05
    sensor_range: float = 3.0
    sensor_fov_deg: float = 180
    sensor_num_rays: int = 21
    sensor_noise_std: float = 0.01
    bump_enabled: bool = True
    safety_stop_distance: float = 0.25

class Simulator:
    def __init__(self, env: Environment, robot: Robot, cfg: SimulatorConfig = SimulatorConfig()):
        self.env = env
        self.robot = robot
        self.cfg = cfg
        self.time: float = 0.0
        self.log: List[Dict] = []
        self.controller: Optional[Callable[[SensorReading], Tuple[float, float]]] = None
        self.step_callback: Optional[Callable[[Simulator], None]] = None

    def step(self, control: Optional[Tuple[float, float]] = None) -> SensorReading:
        obs = self.sense()

        # safety: immediate stop on bump
        if self.cfg.bump_enabled and obs.bump:
            applied_v, applied_omega = 0.0, 0.0
        else:
            if control is None and self.controller is not None:
                applied_v, applied_omega = self.controller(obs)
            elif control is not None:
                applied_v, applied_omega = control
            else:
                applied_v, applied_omega = self.robot.v, self.robot.omega

        # clamp
        applied_v = clamp(applied_v, -self.robot.max_speed, self.robot.max_speed)
        applied_omega = clamp(applied_omega, -self.robot.max_omega, self.robot.max_omega)

        # safety stop based on min distance
        min_dist = min(obs.distances) if obs.distances else float('inf')
        if min_dist < self.cfg.safety_stop_distance:
            applied_v = 0.0

        # integrate
        self.robot.v = applied_v
        self.robot.omega = applied_omega
        self._integrate(self.cfg.dt)
        self.time += self.cfg.dt

        # log snapshot
        self.log.append({
            't': self.time,
            'pose': self.robot_pose(),
            'v': self.robot.v,
            'omega': self.robot.omega,
            'min_dist': min_dist if obs.distances else None,
            'bump': obs.bump,
        })

        # callback
        if self.step_callback:
            try:
                self.step_callback(self)
            except Exception:
                pass

        return obs

    def _integrate(self, dt: float):
        dx = self.robot.v * math.cos(self.robot.theta) * dt
        dy = self.robot.v * math.sin(self.robot.theta) * dt
        dtheta = self.robot.omega * dt
        newx = self.robot.x + dx
        newy = self.robot.y + dy

        # collision check with environment boundaries and obstacles
        if not self.env.is_free(newx, newy):
            # collision: back off, stop motion but still update orientation
            self.robot.v = 0.0
            self.robot.omega = 0.0
            # no position change (simple response)
            self.robot.theta = (self.robot.theta + dtheta) % (2 * math.pi)
            return True

        # commit
        self.robot.x = newx
        self.robot.y = newy
        self.robot.theta = (self.robot.theta + dtheta) % (2 * math.pi)
        return False

    def sense(self) -> SensorReading:
        cfg = self.cfg
        rays = []
        half_fov = math.radians(cfg.sensor_fov_deg) / 2.0
        for i in range(cfg.sensor_num_rays):
            a = -half_fov + i * (2 * half_fov) / max(1, cfg.sensor_num_rays - 1)
            ang = self.robot.theta + a
            dist = self._ray_cast(self.robot.x, self.robot.y, ang, cfg.sensor_range)
            noisy = max(0.0, dist + random.gauss(0, cfg.sensor_noise_std))
            rays.append(noisy)

        bump = False
        if cfg.bump_enabled:
            for o in self.env.obstacles:
                if (self.robot.x - o.x) ** 2 + (self.robot.y - o.y) ** 2 <= (self.robot.radius + o.r) ** 2:
                    bump = True
                    break

        return SensorReading(distances=rays, bump=bump)

    def _ray_cast(self, x: float, y: float, ang: float, max_range: float) -> float:
        step = 0.02
        t = 0.0
        while t < max_range:
            px = x + t * math.cos(ang)
            py = y + t * math.sin(ang)
            if not self.env.is_free(px, py):
                return t
            t += step
        return max_range

    def robot_pose(self):
        return (self.robot.x, self.robot.y, self.robot.theta)

# simple example controller factory
def simple_wall_follow_controller_factory(target_distance: float = 0.5) -> Callable[[SensorReading], Tuple[float, float]]:
    def controller(obs: SensorReading) -> Tuple[float, float]:
        min_dist = min(obs.distances) if obs.distances else float('inf')
        if obs.bump:
            return (0.0, math.pi / 2)
        try:
            idx = obs.distances.index(min_dist)
        except ValueError:
            idx = len(obs.distances) // 2
        center = (len(obs.distances) - 1) / 2
        err = (idx - center) / max(1, center)
        v = 0.5 if min_dist > target_distance else 0.0
        omega = -err * 1.5
        return (v, omega)
    return controller

# streamlit helper (kept minimal)
def run_streamlit_app(sim: Simulator):
    obs = sim.sense()
    status = {
        'time': sim.time,
        'pose': sim.robot_pose(),
        'min_dist': min(obs.distances) if obs.distances else None,
        'bump': obs.bump,
    }
    return status

if __name__ == '__main__':
    # quick test
    env = Environment(width=6.0, height=6.0, obstacles=[Obstacle(3.0, 3.0, 0.6), Obstacle(1.2, 4.0, 0.3)])
    r = Robot(x=0.7, y=0.7, theta=0.0)
    sim = Simulator(env, r)
    sim.controller = simple_wall_follow_controller_factory(0.5)
    print('Starting simulation...')
    for i in range(100):
        obs = sim.step()
        if i % 10 == 0:
            print(f"t={sim.time:.2f} pose={sim.robot_pose()} min_dist={min(obs.distances):.2f} bump={obs.bump}")
        if obs.bump:
            print('BUMP detected -- stopping')
            break
    print('Finished')
