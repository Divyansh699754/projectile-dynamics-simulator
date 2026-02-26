"""
Aerodynamic Drag Model
======================
Mach-dependent drag coefficients (Cd) for standard projectile geometries.

Implements Cd vs Mach curves from published data:
- Flat-nosed cylinder (bluff body)
- Sphere
- Ogive nose (standard military/engineering projectile)
- Ogive + boat-tail (optimized long-range projectile)

Data points are interpolated from:
- McCoy, "Modern Exterior Ballistics" (2012)
- Hoerner, "Fluid Dynamic Drag" (1965)
- NACA / NASA technical reports for standard body shapes

The transonic regime (Mach 0.8–1.2) shows the characteristic drag rise
that dominates projectile performance.
"""

import numpy as np
from scipy.interpolate import interp1d


# ══════════════════════════════════════════════════════════════════════════
#  Cd vs Mach tables — (Mach, Cd) pairs from published references
# ══════════════════════════════════════════════════════════════════════════

# Flat-nosed cylinder — highest drag, massive transonic spike
FLAT_NOSE_DATA = {
    'name': 'Flat-Nose Cylinder',
    'color': '#e74c3c',
    'linestyle': '-',
    'mach': [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    'cd':   [0.85, 0.85, 0.87, 0.92, 1.05, 1.15, 1.30, 1.35, 1.32, 1.22, 1.10, 1.02, 0.96, 0.88],
}

# Sphere — classic reference shape
SPHERE_DATA = {
    'name': 'Sphere',
    'color': '#3498db',
    'linestyle': '--',
    'mach': [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    'cd':   [0.47, 0.47, 0.48, 0.50, 0.58, 0.70, 0.92, 0.96, 0.94, 0.88, 0.80, 0.74, 0.70, 0.65],
}

# Ogive nose — standard pointed projectile (e.g., 7-caliber ogive, no boat-tail)
OGIVE_DATA = {
    'name': 'Ogive Nose',
    'color': '#2ecc71',
    'linestyle': '-.',
    'mach': [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    'cd':   [0.35, 0.35, 0.36, 0.40, 0.50, 0.60, 0.72, 0.68, 0.64, 0.55, 0.48, 0.44, 0.42, 0.39],
}

# Ogive + boat-tail — realistic long-range projectile (includes form factor for
# real projectile effects: rotating bands, bourrelet, surface roughness)
# Tuned to approximate G7 ballistic standard / McCoy M107 data
OGIVE_BOATTAIL_DATA = {
    'name': 'Ogive + Boat-Tail',
    'color': '#f39c12',
    'linestyle': ':',
    'mach': [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    'cd':   [0.30, 0.30, 0.31, 0.34, 0.44, 0.55, 0.62, 0.59, 0.56, 0.48, 0.42, 0.39, 0.37, 0.34],
}

# Collect all geometries
ALL_GEOMETRIES = {
    'flat_nose': FLAT_NOSE_DATA,
    'sphere': SPHERE_DATA,
    'ogive': OGIVE_DATA,
    'ogive_boattail': OGIVE_BOATTAIL_DATA,
}


# ══════════════════════════════════════════════════════════════════════════
#  Interpolation-based Cd lookup
# ══════════════════════════════════════════════════════════════════════════

class DragModel:
    """
    Drag coefficient model for a specific projectile geometry.

    Uses cubic spline interpolation over the Cd vs Mach data,
    with linear extrapolation beyond the table limits.
    """

    def __init__(self, geometry_key: str = 'ogive_boattail'):
        """
        Parameters
        ----------
        geometry_key : str
            One of 'flat_nose', 'sphere', 'ogive', 'ogive_boattail'
        """
        if geometry_key not in ALL_GEOMETRIES:
            raise ValueError(
                f"Unknown geometry '{geometry_key}'. "
                f"Available: {list(ALL_GEOMETRIES.keys())}"
            )

        data = ALL_GEOMETRIES[geometry_key]
        self.name = data['name']
        self.color = data['color']
        self.linestyle = data['linestyle']
        self.geometry_key = geometry_key

        mach_arr = np.array(data['mach'])
        cd_arr = np.array(data['cd'])

        # Cubic interpolation with linear extrapolation
        self._interp = interp1d(
            mach_arr, cd_arr,
            kind='cubic',
            fill_value='extrapolate',
            assume_sorted=True,
        )

    def cd(self, mach: float) -> float:
        """Return drag coefficient at the given Mach number."""
        mach = max(mach, 0.0)
        cd_val = float(self._interp(mach))
        return max(cd_val, 0.01)  # physical floor

    def cd_array(self, mach_array: np.ndarray) -> np.ndarray:
        """Vectorized Cd lookup."""
        mach_array = np.clip(mach_array, 0.0, None)
        cd_vals = self._interp(mach_array)
        return np.clip(cd_vals, 0.01, None)


def drag_force(velocity_rel: np.ndarray, rho: float, cd: float,
               area: float) -> np.ndarray:
    """
    Compute aerodynamic drag force vector (N).

    F_drag = -½ ρ |v_rel|² Cd A v̂_rel

    Parameters
    ----------
    velocity_rel : np.ndarray
        Velocity relative to air mass [vx, vy, vz] (m/s)
    rho : float
        Air density (kg/m³)
    cd : float
        Drag coefficient (dimensionless)
    area : float
        Reference cross-sectional area (m²)

    Returns
    -------
    np.ndarray
        Drag force vector [Fx, Fy, Fz] (N)
    """
    v_mag = np.linalg.norm(velocity_rel)
    if v_mag < 1e-10:
        return np.zeros(3)

    v_hat = velocity_rel / v_mag
    F_mag = 0.5 * rho * v_mag ** 2 * cd * area
    return -F_mag * v_hat


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Drag Model — Cd vs Mach for All Geometries")
    print("=" * 50)

    mach_range = np.linspace(0, 4.0, 500)

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, data in ALL_GEOMETRIES.items():
        model = DragModel(key)
        cd_vals = model.cd_array(mach_range)
        ax.plot(mach_range, cd_vals, label=model.name,
                color=data['color'], linestyle=data['linestyle'], linewidth=2)

    ax.set_xlabel('Mach Number', fontsize=12)
    ax.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
    ax.set_title('Cd vs Mach — Standard Projectile Geometries', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 1.5)
    plt.tight_layout()
    plt.savefig('outputs/cd_vs_mach.png', dpi=150)
    print("Saved: outputs/cd_vs_mach.png")
