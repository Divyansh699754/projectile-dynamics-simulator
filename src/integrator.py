"""
Numerical Integration Engine
=============================
Implements two time-stepping methods for trajectory computation:

1. **Euler Method** (1st order) — Simple, fast, but accumulates error.
2. **Runge-Kutta 4th Order (RK4)** — Standard engineering method,
   much more accurate at the same timestep.

Both integrate the equations of motion:
    dx/dt = v
    dv/dt = a(x, v, t)  (from compute_forces)

Output: TrajectoryResult dataclass with full state history.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Callable

from .projectile import Projectile, LaunchConditions, compute_forces


@dataclass
class TrajectoryState:
    """Snapshot of projectile state at one instant."""
    time: float
    position: np.ndarray   # [x, y, z]
    velocity: np.ndarray   # [vx, vy, vz]
    mach: float
    cd: float
    density: float


@dataclass
class TrajectoryResult:
    """Complete trajectory output."""
    projectile: Projectile
    conditions: LaunchConditions
    method: str               # 'euler' or 'rk4'
    dt: float                 # timestep used

    # Arrays — each has shape (N,)
    time: np.ndarray
    x: np.ndarray             # downrange
    y: np.ndarray             # altitude
    z: np.ndarray             # crossrange
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    speed: np.ndarray
    mach_history: np.ndarray
    cd_history: np.ndarray
    density_history: np.ndarray

    @property
    def range_total(self) -> float:
        """Total horizontal range at impact (m)."""
        return self.x[-1]

    @property
    def max_altitude(self) -> float:
        """Maximum altitude reached (m)."""
        return float(np.max(self.y))

    @property
    def flight_time(self) -> float:
        """Total flight time (s)."""
        return self.time[-1]

    @property
    def impact_velocity(self) -> float:
        """Speed at impact (m/s)."""
        return self.speed[-1]

    @property
    def impact_angle_deg(self) -> float:
        """Angle of descent at impact (degrees below horizontal)."""
        vx_f = self.vx[-1]
        vy_f = self.vy[-1]
        v_horiz = np.sqrt(vx_f**2 + self.vz[-1]**2)
        return float(np.degrees(np.arctan2(-vy_f, v_horiz)))

    @property
    def lateral_drift(self) -> float:
        """Total crossrange drift at impact (m)."""
        return self.z[-1]

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"╔══════════════════════════════════════════════════════╗",
            f"║  TRAJECTORY SUMMARY — {self.projectile.name:<30s} ║",
            f"╠══════════════════════════════════════════════════════╣",
            f"║  Geometry     : {self.projectile.drag_model.name:<36s} ║",
            f"║  Method       : {self.method.upper():<36s} ║",
            f"║  Timestep     : {self.dt:<36.4f} ║",
            f"╠══════════════════════════════════════════════════════╣",
            f"║  Launch vel   : {self.conditions.velocity:>10.1f} m/s{'':<22s} ║",
            f"║  Elevation    : {self.conditions.elevation_deg:>10.1f} °{'':<24s} ║",
            f"╠══════════════════════════════════════════════════════╣",
            f"║  Range        : {self.range_total:>10.1f} m  ({self.range_total/1000:>7.2f} km){'':<6s} ║",
            f"║  Max altitude : {self.max_altitude:>10.1f} m  ({self.max_altitude/1000:>7.2f} km){'':<6s} ║",
            f"║  Flight time  : {self.flight_time:>10.2f} s{'':<24s} ║",
            f"║  Impact vel   : {self.impact_velocity:>10.1f} m/s{'':<22s} ║",
            f"║  Impact angle : {self.impact_angle_deg:>10.1f} °{'':<24s} ║",
            f"║  Lateral drift: {self.lateral_drift:>10.1f} m{'':<24s} ║",
            f"╚══════════════════════════════════════════════════════╝",
        ]
        return '\n'.join(lines)


def _record_state(t, pos, vel, proj, cond):
    """Helper to compute and record derived quantities."""
    from .atmosphere import isa_density, speed_of_sound
    alt = max(pos[1], 0.0)
    spd = np.linalg.norm(vel)
    a = speed_of_sound(alt)
    m = spd / a if a > 0 else 0.0
    cd = proj.drag_model.cd(m)
    rho = isa_density(alt)
    return t, pos.copy(), vel.copy(), spd, m, cd, rho


def simulate_euler(projectile: Projectile, conditions: LaunchConditions,
                   dt: float = 0.01, max_time: float = 300.0,
                   enable_coriolis: bool = True) -> TrajectoryResult:
    """
    Forward Euler integration.

    x_{n+1} = x_n + v_n * dt
    v_{n+1} = v_n + a(x_n, v_n) * dt
    """
    pos = conditions.initial_position()
    vel = conditions.initial_velocity_vector()
    t = 0.0

    history = [_record_state(t, pos, vel, projectile, conditions)]

    while t < max_time:
        acc = compute_forces(pos, vel, projectile, conditions, enable_coriolis)

        pos = pos + vel * dt
        vel = vel + acc * dt
        t += dt

        history.append(_record_state(t, pos, vel, projectile, conditions))

        # Stop when projectile hits ground (below launch altitude)
        if pos[1] < conditions.launch_altitude and t > dt:
            break

    return _build_result(history, projectile, conditions, 'euler', dt)


def simulate_rk4(projectile: Projectile, conditions: LaunchConditions,
                 dt: float = 0.1, max_time: float = 300.0,
                 enable_coriolis: bool = True) -> TrajectoryResult:
    """
    4th-order Runge-Kutta integration.

    Much more accurate than Euler at the same timestep.
    Standard method used in real ballistic computation codes.
    """
    pos = conditions.initial_position()
    vel = conditions.initial_velocity_vector()
    t = 0.0

    history = [_record_state(t, pos, vel, projectile, conditions)]

    def accel(p, v):
        return compute_forces(p, v, projectile, conditions, enable_coriolis)

    while t < max_time:
        # RK4 stages
        k1v = accel(pos, vel)
        k1x = vel

        k2v = accel(pos + 0.5 * dt * k1x, vel + 0.5 * dt * k1v)
        k2x = vel + 0.5 * dt * k1v

        k3v = accel(pos + 0.5 * dt * k2x, vel + 0.5 * dt * k2v)
        k3x = vel + 0.5 * dt * k2v

        k4v = accel(pos + dt * k3x, vel + dt * k3v)
        k4x = vel + dt * k3v

        pos = pos + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        vel = vel + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        t += dt

        history.append(_record_state(t, pos, vel, projectile, conditions))

        if pos[1] < conditions.launch_altitude and t > dt:
            break

    return _build_result(history, projectile, conditions, 'rk4', dt)


def _build_result(history, projectile, conditions, method, dt):
    """Convert history list to TrajectoryResult."""
    times, positions, velocities, speeds, machs, cds, rhos = zip(*history)

    positions = np.array(positions)
    velocities = np.array(velocities)

    return TrajectoryResult(
        projectile=projectile,
        conditions=conditions,
        method=method,
        dt=dt,
        time=np.array(times),
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        vx=velocities[:, 0],
        vy=velocities[:, 1],
        vz=velocities[:, 2],
        speed=np.array(speeds),
        mach_history=np.array(machs),
        cd_history=np.array(cds),
        density_history=np.array(rhos),
    )
