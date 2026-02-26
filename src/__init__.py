"""
Projectile Dynamics Simulator with Atmospheric Modeling
=======================================================
A computational tool that models the complete flight path of a projectile
from launch to impact, incorporating all major physical forces:
  - Gravity
  - Mach-dependent aerodynamic drag
  - ISA atmospheric model (variable density, pressure, temperature)
  - Wind effects (head/tail/cross)
  - Coriolis effect (Earth's rotation)

Supports multiple projectile geometries with published Cd vs Mach data,
both Euler and RK4 numerical integration, and validation against
McCoy's "Modern Exterior Ballistics" firing tables.

Author: Divyansh
Date: February 2026
"""

from .atmosphere import (
    isa_temperature, isa_pressure, isa_density,
    speed_of_sound, mach_number, isa_profile,
)
from .drag_model import DragModel, drag_force, ALL_GEOMETRIES
from .projectile import Projectile, LaunchConditions, compute_forces
from .integrator import simulate_euler, simulate_rk4, TrajectoryResult
from .validation import (
    validate_against_reference, run_all_validations,
    REFERENCE_M107, REFERENCE_M1,
)
from .visualization import (
    plot_trajectory, plot_geometry_comparison, plot_cd_vs_mach,
    plot_atmosphere, plot_dashboard, plot_euler_vs_rk4,
    plot_validation, create_trajectory_animation,
)

__version__ = "1.0.0"
__all__ = [
    'Projectile', 'LaunchConditions', 'TrajectoryResult',
    'simulate_euler', 'simulate_rk4',
    'DragModel', 'ALL_GEOMETRIES',
    'isa_temperature', 'isa_pressure', 'isa_density',
    'speed_of_sound', 'mach_number',
    'validate_against_reference', 'run_all_validations',
    'plot_trajectory', 'plot_geometry_comparison', 'plot_cd_vs_mach',
    'plot_atmosphere', 'plot_dashboard', 'plot_euler_vs_rk4',
    'plot_validation', 'create_trajectory_animation',
]
