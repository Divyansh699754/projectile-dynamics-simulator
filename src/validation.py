"""
Validation Against Published Data
===================================
Compares simulator output against reference firing tables from:
  - McCoy, R.L. "Modern Exterior Ballistics" (2012)
  - US Army BRL firing tables

Standard reference projectile:
  - 155mm M107 HE (ogive + boat-tail)
  - Mass: 43.2 kg
  - Diameter: 0.155 m
  - Muzzle velocity: 684 m/s (Charge 7, M4A2)

Reference data points (range vs elevation) extracted from
published firing tables for sea-level standard atmosphere.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .projectile import Projectile, LaunchConditions
from .integrator import simulate_rk4, TrajectoryResult


# ══════════════════════════════════════════════════════════════════════════
#  Reference data — 155mm M107 firing table (simplified from McCoy)
# ══════════════════════════════════════════════════════════════════════════

# (elevation_deg, expected_range_m, expected_max_alt_m, expected_time_of_flight_s)
# These are approximate values from published tables for validation
REFERENCE_M107 = {
    'name': '155mm M107 HE',
    'mass': 43.2,
    'diameter': 0.155,
    'muzzle_velocity': 684.0,
    'geometry': 'ogive_boattail',
    'firing_table': [
        # (elev°,  range_m,  max_alt_m, tof_s)
        (10.0,    5800,     280,        14.5),
        (15.0,    8200,     570,        19.5),
        (20.0,   10200,     950,        24.5),
        (25.0,   11800,    1400,        29.5),
        (30.0,   13000,    1950,        34.5),
        (35.0,   13800,    2550,        39.5),
        (40.0,   14200,    3200,        44.5),
        (45.0,   14300,    3850,        49.0),
        (50.0,   13900,    4500,        53.5),
    ],
}

# Secondary validation — 105mm M1 round
REFERENCE_M1 = {
    'name': '105mm M1 HE',
    'mass': 14.97,
    'diameter': 0.105,
    'muzzle_velocity': 472.0,
    'geometry': 'ogive_boattail',
    'firing_table': [
        # (elev°,  range_m,  max_alt_m, tof_s)
        (15.0,    4200,     300,       13.0),
        (30.0,    7200,    1000,       23.0),
        (45.0,    8200,    2000,       32.0),
    ],
}


@dataclass
class ValidationResult:
    """Result of one validation comparison."""
    elevation_deg: float
    ref_range: float        # reference range (m)
    sim_range: float        # simulated range (m)
    range_error_pct: float  # % error
    ref_max_alt: float
    sim_max_alt: float
    alt_error_pct: float
    ref_tof: float
    sim_tof: float
    tof_error_pct: float


def validate_against_reference(reference: dict, dt: float = 0.05,
                               verbose: bool = True) -> List[ValidationResult]:
    """
    Run the simulator at each firing table condition and compare
    against the published reference data.

    Returns list of ValidationResult for each elevation.
    """
    proj = Projectile(
        name=reference['name'],
        mass=reference['mass'],
        diameter=reference['diameter'],
        geometry=reference['geometry'],
    )

    results = []

    if verbose:
        print(f"\n{'='*75}")
        print(f"  VALIDATION: {reference['name']}")
        print(f"  Muzzle velocity: {reference['muzzle_velocity']} m/s")
        print(f"  Diameter: {reference['diameter']*1000:.0f} mm | Mass: {reference['mass']} kg")
        print(f"{'='*75}")
        print(f"{'Elev°':>6} {'Ref R (m)':>10} {'Sim R (m)':>10} {'Err %':>7} "
              f"{'Ref Alt':>9} {'Sim Alt':>9} {'Err %':>7} "
              f"{'Ref ToF':>8} {'Sim ToF':>8} {'Err %':>7}")
        print("-" * 75)

    for elev, ref_range, ref_alt, ref_tof in reference['firing_table']:
        cond = LaunchConditions(
            velocity=reference['muzzle_velocity'],
            elevation_deg=elev,
            azimuth_deg=0.0,
            launch_altitude=0.0,
            latitude_deg=30.0,
            wind_x=0.0,
            wind_y=0.0,
            wind_z=0.0,
        )

        traj = simulate_rk4(proj, cond, dt=dt, enable_coriolis=False)

        range_err = 100.0 * (traj.range_total - ref_range) / ref_range
        alt_err = 100.0 * (traj.max_altitude - ref_alt) / ref_alt
        tof_err = 100.0 * (traj.flight_time - ref_tof) / ref_tof

        vr = ValidationResult(
            elevation_deg=elev,
            ref_range=ref_range,
            sim_range=traj.range_total,
            range_error_pct=range_err,
            ref_max_alt=ref_alt,
            sim_max_alt=traj.max_altitude,
            alt_error_pct=alt_err,
            ref_tof=ref_tof,
            sim_tof=traj.flight_time,
            tof_error_pct=tof_err,
        )
        results.append(vr)

        if verbose:
            print(f"{elev:>6.0f} {ref_range:>10.0f} {traj.range_total:>10.0f} "
                  f"{range_err:>+7.1f} "
                  f"{ref_alt:>9.0f} {traj.max_altitude:>9.0f} {alt_err:>+7.1f} "
                  f"{ref_tof:>8.1f} {traj.flight_time:>8.1f} {tof_err:>+7.1f}")

    if verbose:
        avg_range_err = np.mean([abs(r.range_error_pct) for r in results])
        avg_alt_err = np.mean([abs(r.alt_error_pct) for r in results])
        avg_tof_err = np.mean([abs(r.tof_error_pct) for r in results])
        print("-" * 75)
        print(f"  Mean absolute errors — Range: {avg_range_err:.1f}% | "
              f"Altitude: {avg_alt_err:.1f}% | Time: {avg_tof_err:.1f}%")
        status = "✓ PASS" if avg_range_err < 15 else "✗ NEEDS TUNING"
        print(f"  Status: {status}")
        print(f"{'='*75}\n")

    return results


def run_all_validations(verbose: bool = True):
    """Run validation against all available reference datasets."""
    all_results = {}
    for ref_data in [REFERENCE_M107, REFERENCE_M1]:
        results = validate_against_reference(ref_data, verbose=verbose)
        all_results[ref_data['name']] = results
    return all_results


if __name__ == "__main__":
    run_all_validations(verbose=True)
