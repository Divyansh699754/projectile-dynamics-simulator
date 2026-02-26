# Projectile Dynamics Simulator with Atmospheric Modeling

A computational tool that models the complete flight path of a projectile from launch to impact, incorporating every major physical force that acts on a body moving through the atmosphere at high velocity.

> In reality, a projectile doesn't follow a clean parabola. Air resistance steals energy. Wind pushes it sideways. Air gets thinner as it climbs. Earth rotates underneath it during flight. This simulator accounts for **all of that**.

---

## Physics Engine

### Forces Modeled

| Force | Model | Notes |
|-------|-------|-------|
| **Gravity** | Constant 9.80665 m/s² | Baseline downward acceleration |
| **Aerodynamic Drag** | F = ½ρv²CdA | Quadratic drag with Mach-dependent Cd |
| **Atmospheric Variation** | ISA 1976 | Density, pressure, temperature vs altitude |
| **Wind** | 3D vector field | Head/tail/crosswind components |
| **Coriolis Effect** | Earth rotation model | Function of latitude, azimuth, flight time |

### Drag Coefficient Model

The drag coefficient is **not constant** — it varies with Mach number. The simulator implements full Cd vs Mach curves from published data (McCoy, Hoerner) for four standard geometries:

- **Flat-Nose Cylinder** — Highest drag, massive transonic spike
- **Sphere** — Classic reference shape
- **Ogive Nose** — Standard pointed projectile shape
- **Ogive + Boat-Tail** — Optimized long-range design (lowest drag)

### International Standard Atmosphere (ISA)

Air density, pressure, temperature, and speed of sound are computed as functions of altitude using the ISA 1976 model:
- Troposphere (0–11 km): Linear temperature lapse at −6.5 °C/km
- Stratosphere (11–20 km): Isothermal at 216.65 K
- Upper layers: Extended model to 32+ km

### Numerical Integration

Two methods implemented for comparison:

1. **Euler Method** (1st order) — Simple, intuitive, accumulates error
2. **Runge-Kutta 4th Order (RK4)** — Industry-standard accuracy

### Validation

Validated against published firing table data from McCoy's *Modern Exterior Ballistics*:
- **155mm M107 HE** — 9 elevation angles (10°–50°)
- **105mm M1 HE** — 3 elevation angles

---

## Project Structure

```
projectile-dynamics-simulator/
├── main.py                  # Main runner — executes full simulation pipeline
├── requirements.txt         # Python dependencies
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py          # Package exports
│   ├── atmosphere.py        # ISA atmospheric model
│   ├── drag_model.py        # Cd vs Mach curves for 4 geometries
│   ├── projectile.py        # Projectile definition & force computation
│   ├── integrator.py        # Euler & RK4 numerical integrators
│   ├── validation.py        # Comparison against McCoy firing tables
│   └── visualization.py     # All plotting & animation functions
├── outputs/                 # Generated plots and animations
│   └── .gitkeep
├── tests/
│   └── test_physics.py      # Unit tests for core physics
└── docs/
    └── physics_notes.md     # Detailed physics documentation
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Divyansh699754/projectile-dynamics-simulator.git
cd projectile-dynamics-simulator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Simulation

```bash
python main.py
```

This runs all 10 phases and generates all output plots in `outputs/`.

For a faster run (skips GIF animation):

```bash
python main.py --quick
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Outputs

The simulator generates these visualizations:

| File | Description |
|------|-------------|
| `01_atmosphere_profile.png` | ISA model: T, P, ρ, speed of sound vs altitude |
| `02_cd_vs_mach.png` | Drag coefficient curves for all 4 geometries |
| `03_reference_trajectory.png` | Single trajectory (155mm at 45°) |
| `04_geometry_comparison.png` | 4-panel comparison of all shapes |
| `05_euler_vs_rk4.png` | Numerical method accuracy comparison |
| `06_validation_m107.png` | 155mm M107 validated against McCoy |
| `06b_validation_m1.png` | 105mm M1 validated against McCoy |
| `07_wind_effects.png` | Head/tail/crosswind effects |
| `08_coriolis_effect.png` | Earth rotation deflection |
| `09_dashboard.png` | Full flight data dashboard |
| `10_trajectory_animation.gif` | Animated projectile flight |

---

## Using as a Library

```python
from src import Projectile, LaunchConditions, simulate_rk4

# Define your projectile
proj = Projectile(
    name="My Projectile",
    mass=10.0,           # kg
    diameter=0.105,      # m
    geometry='ogive',    # 'flat_nose', 'sphere', 'ogive', 'ogive_boattail'
)

# Set launch conditions
cond = LaunchConditions(
    velocity=500.0,       # m/s
    elevation_deg=30.0,   # degrees
    wind_x=5.0,           # headwind (m/s)
    wind_z=2.0,           # crosswind (m/s)
    latitude_deg=45.0,    # for Coriolis
)

# Simulate
result = simulate_rk4(proj, cond, dt=0.05)

# Results
print(result.summary())
print(f"Range: {result.range_total/1000:.2f} km")
print(f"Max altitude: {result.max_altitude:.0f} m")
print(f"Flight time: {result.flight_time:.1f} s")
```

---

## Technical Skills Demonstrated

- **Classical Mechanics** — Newtonian force balance, projectile motion
- **Aerodynamics** — Mach-dependent drag, transonic phenomena, geometry effects
- **Atmospheric Science** — ISA model, altitude-dependent properties
- **Numerical Methods** — Euler vs RK4, timestep convergence, error analysis
- **Python Engineering** — Modular OOP design, NumPy/SciPy/Matplotlib
- **Data Visualization** — Publication-quality plots, dashboards, animations
- **Validation** — Comparison against experimental reference data

---

## References

1. McCoy, R.L. *Modern Exterior Ballistics* (2012) — Firing table data
2. Hoerner, S.F. *Fluid-Dynamic Drag* (1965) — Drag coefficient data
3. U.S. Standard Atmosphere, 1976 (NASA-TM-X-74335) — ISA model
4. NACA Technical Reports — Standard body aerodynamic data

---

## License

MIT License — See [LICENSE](LICENSE) for details.
