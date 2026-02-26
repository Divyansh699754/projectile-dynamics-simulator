"""
Projectile Definition & Forces
===============================
Defines the Projectile dataclass and computes all physical forces:
  - Gravity
  - Aerodynamic drag (Mach-dependent)
  - Wind effects (velocity relative to air mass)
  - Coriolis effect (Earth's rotation)

Coordinate system:
  x = downrange (horizontal)
  y = altitude  (vertical, up positive)
  z = crossrange (lateral, right positive looking downrange)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .atmosphere import isa_density, speed_of_sound, GRAVITY
from .drag_model import DragModel, drag_force


# ── Earth rotation parameters ─────────────────────────────────────────────
EARTH_ROTATION_RATE = 7.2921e-5  # rad/s


@dataclass
class Projectile:
    """
    Defines a projectile with its physical and aerodynamic properties.
    """
    name: str = "Standard Projectile"
    mass: float = 10.0               # kg
    diameter: float = 0.155           # m  (155 mm — standard artillery caliber)
    geometry: str = 'ogive_boattail'  # drag profile key
    drag_model: Optional[DragModel] = field(default=None, repr=False)

    def __post_init__(self):
        self.area = np.pi * (self.diameter / 2) ** 2  # reference area (m²)
        if self.drag_model is None:
            self.drag_model = DragModel(self.geometry)

    @property
    def caliber(self) -> float:
        """Diameter in mm."""
        return self.diameter * 1000


@dataclass
class LaunchConditions:
    """
    Complete specification of launch parameters.
    """
    velocity: float = 800.0           # m/s  muzzle velocity
    elevation_deg: float = 45.0       # degrees above horizontal
    azimuth_deg: float = 0.0          # degrees from North (0=N, 90=E)
    launch_altitude: float = 0.0      # m ASL
    latitude_deg: float = 30.0        # degrees (for Coriolis)

    # Wind components (m/s) — velocity of the air mass
    # Positive wind_x = air moving in +x (tailwind), negative = headwind
    wind_x: float = 0.0              # tailwind (+) / headwind (-)
    wind_y: float = 0.0              # vertical wind (usually 0)
    wind_z: float = 0.0              # crosswind from right (+) / left (-)

    def initial_velocity_vector(self) -> np.ndarray:
        """
        Convert launch speed + angles to [vx, vy, vz] vector.
        """
        elev = np.radians(self.elevation_deg)
        azim = np.radians(self.azimuth_deg)

        vx = self.velocity * np.cos(elev) * np.cos(azim)
        vy = self.velocity * np.sin(elev)
        vz = self.velocity * np.cos(elev) * np.sin(azim)
        return np.array([vx, vy, vz])

    def initial_position(self) -> np.ndarray:
        """Starting position [x, y, z]."""
        return np.array([0.0, self.launch_altitude, 0.0])

    @property
    def wind_vector(self) -> np.ndarray:
        return np.array([self.wind_x, self.wind_y, self.wind_z])


def compute_forces(position: np.ndarray, velocity: np.ndarray,
                   projectile: Projectile, conditions: LaunchConditions,
                   enable_coriolis: bool = True) -> np.ndarray:
    """
    Compute total force vector acting on the projectile.

    Parameters
    ----------
    position : [x, y, z] in meters
    velocity : [vx, vy, vz] in m/s (ground-frame)
    projectile : Projectile instance
    conditions : LaunchConditions instance
    enable_coriolis : bool

    Returns
    -------
    acceleration : np.ndarray [ax, ay, az] in m/s²
    """
    altitude = max(position[1], 0.0)  # clip to ground level for atmosphere

    # ── 1. Gravity ────────────────────────────────────────────────────────
    a_gravity = np.array([0.0, -GRAVITY, 0.0])

    # ── 2. Aerodynamic drag ───────────────────────────────────────────────
    # Velocity relative to air mass (subtract wind from projectile velocity)
    v_rel = velocity - conditions.wind_vector

    v_mag = np.linalg.norm(v_rel)
    rho = isa_density(altitude)
    a_sound = speed_of_sound(altitude)
    mach = v_mag / a_sound if a_sound > 0 else 0.0

    cd = projectile.drag_model.cd(mach)
    F_drag = drag_force(v_rel, rho, cd, projectile.area)
    a_drag = F_drag / projectile.mass

    # ── 3. Coriolis acceleration ──────────────────────────────────────────
    a_coriolis = np.zeros(3)
    if enable_coriolis:
        lat = np.radians(conditions.latitude_deg)
        # Earth's angular velocity vector in local frame
        # (simplified: x=North, y=Up, z=East)
        omega_earth = EARTH_ROTATION_RATE * np.array([
            np.cos(lat),   # North component
            np.sin(lat),   # Up component (vertical)
            0.0
        ])
        # Coriolis acceleration = -2 (Ω × v)
        a_coriolis = -2.0 * np.cross(omega_earth, velocity)

    # ── Total acceleration ────────────────────────────────────────────────
    return a_gravity + a_drag + a_coriolis
