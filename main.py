#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  PROJECTILE DYNAMICS SIMULATOR — Main Runner
═══════════════════════════════════════════════════════════════════════════════

  Executes the complete simulation pipeline:
    1. Atmospheric model verification
    2. Cd vs Mach curve generation
    3. Single trajectory simulation (RK4)
    4. Geometry comparison (all 4 shapes)
    5. Euler vs RK4 accuracy comparison
    6. Validation against McCoy's firing tables
    7. Full dashboard generation
    8. Animated trajectory GIF
    9. Wind & Coriolis effect demonstrations

  All outputs saved to outputs/ directory.

  Usage:
    python main.py              # Run everything
    python main.py --quick      # Skip animation (faster)

  Author: Divyansh
  Date: February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.atmosphere import isa_temperature, isa_pressure, isa_density, speed_of_sound
from src.drag_model import DragModel, ALL_GEOMETRIES
from src.projectile import Projectile, LaunchConditions
from src.integrator import simulate_euler, simulate_rk4
from src.validation import validate_against_reference, REFERENCE_M107, REFERENCE_M1
from src.visualization import (
    plot_trajectory, plot_geometry_comparison, plot_cd_vs_mach,
    plot_atmosphere, plot_dashboard, plot_euler_vs_rk4,
    plot_validation, create_trajectory_animation,
    ensure_output_dir,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗       ║
║     ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝      ║
║     ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║          ║
║     ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║          ║
║     ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║          ║
║     ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝         ║
║                                                                       ║
║     PROJECTILE DYNAMICS SIMULATOR WITH ATMOSPHERIC MODELING           ║
║     ─────────────────────────────────────────────────────             ║
║     Full physics: Gravity · Drag(Mach) · ISA · Wind · Coriolis       ║
║     Methods: Euler · RK4 │ Validated against McCoy (2012)            ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    start_time = time.time()
    quick = '--quick' in sys.argv

    banner()
    out = ensure_output_dir('outputs')

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Atmospheric Model Verification
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 1: ISA Atmospheric Model")
    print(f"  {'Alt (m)':>8} {'T (K)':>8} {'P (Pa)':>10} {'ρ (kg/m³)':>11} {'a (m/s)':>8}")
    for h in [0, 1000, 5000, 10000, 11000, 15000, 20000]:
        T = isa_temperature(h)
        P = isa_pressure(h)
        rho = isa_density(h)
        a = speed_of_sound(h)
        print(f"  {h:>8} {T:>8.2f} {P:>10.0f} {rho:>11.5f} {a:>8.1f}")

    fig_atm = plot_atmosphere(save_path=f'{out}/01_atmosphere_profile.png')
    plt.close(fig_atm)
    print(f"\n  ✓ Saved: {out}/01_atmosphere_profile.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: Drag Coefficient Curves
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 2: Cd vs Mach Curves")
    fig_cd = plot_cd_vs_mach(save_path=f'{out}/02_cd_vs_mach.png')
    plt.close(fig_cd)
    print(f"  ✓ Saved: {out}/02_cd_vs_mach.png")

    for key in ALL_GEOMETRIES:
        model = DragModel(key)
        print(f"  {model.name:<25s}  Cd @ M0.5={model.cd(0.5):.3f}  "
              f"Cd @ M1.0={model.cd(1.0):.3f}  Cd @ M2.0={model.cd(2.0):.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: Single Trajectory — Ogive+Boat-Tail Reference
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 3: Reference Trajectory (155mm Ogive+Boat-Tail)")

    proj = Projectile(
        name="155mm Artillery Round",
        mass=43.2,
        diameter=0.155,
        geometry='ogive_boattail',
    )
    cond = LaunchConditions(
        velocity=684.0,
        elevation_deg=45.0,
        azimuth_deg=0.0,
        launch_altitude=0.0,
        latitude_deg=30.0,
    )

    result_rk4 = simulate_rk4(proj, cond, dt=0.05)
    print(result_rk4.summary())

    fig_traj = plot_trajectory(result_rk4, save_path=f'{out}/03_reference_trajectory.png')
    plt.close(fig_traj)
    print(f"  ✓ Saved: {out}/03_reference_trajectory.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 4: Geometry Comparison
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 4: Geometry Comparison (Same Launch Conditions)")

    geometry_results = {}
    for key in ALL_GEOMETRIES:
        p = Projectile(
            name=f"Test {ALL_GEOMETRIES[key]['name']}",
            mass=43.2,
            diameter=0.155,
            geometry=key,
        )
        r = simulate_rk4(p, cond, dt=0.05, enable_coriolis=False)
        geometry_results[key] = r
        print(f"  {ALL_GEOMETRIES[key]['name']:<25s}  "
              f"Range: {r.range_total/1000:>7.2f} km  "
              f"Max Alt: {r.max_altitude/1000:>6.2f} km  "
              f"ToF: {r.flight_time:>6.1f} s")

    fig_geo = plot_geometry_comparison(geometry_results,
                                       save_path=f'{out}/04_geometry_comparison.png')
    plt.close(fig_geo)
    print(f"\n  ✓ Saved: {out}/04_geometry_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 5: Euler vs RK4 Accuracy
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 5: Euler vs RK4 Numerical Accuracy")

    dt_test = 0.5  # Large timestep to show differences
    result_euler = simulate_euler(proj, cond, dt=dt_test, enable_coriolis=False)
    result_rk4_compare = simulate_rk4(proj, cond, dt=dt_test, enable_coriolis=False)

    print(f"  Timestep: {dt_test} s")
    print(f"  Euler  — Range: {result_euler.range_total/1000:.2f} km  |  "
          f"Max Alt: {result_euler.max_altitude/1000:.2f} km")
    print(f"  RK4    — Range: {result_rk4_compare.range_total/1000:.2f} km  |  "
          f"Max Alt: {result_rk4_compare.max_altitude/1000:.2f} km")
    print(f"  Δ Range: {(result_euler.range_total - result_rk4_compare.range_total)/1000:+.2f} km")

    fig_evr = plot_euler_vs_rk4(result_euler, result_rk4_compare,
                                 save_path=f'{out}/05_euler_vs_rk4.png')
    plt.close(fig_evr)
    print(f"\n  ✓ Saved: {out}/05_euler_vs_rk4.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 6: Validation Against McCoy's Data
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 6: Validation — 155mm M107 (McCoy)")
    val_results_m107 = validate_against_reference(REFERENCE_M107, dt=0.05)
    fig_val = plot_validation(val_results_m107, REFERENCE_M107,
                              save_path=f'{out}/06_validation_m107.png')
    plt.close(fig_val)
    print(f"  ✓ Saved: {out}/06_validation_m107.png")

    section("PHASE 6b: Validation — 105mm M1")
    val_results_m1 = validate_against_reference(REFERENCE_M1, dt=0.05)
    fig_val2 = plot_validation(val_results_m1, REFERENCE_M1,
                                save_path=f'{out}/06b_validation_m1.png')
    plt.close(fig_val2)
    print(f"  ✓ Saved: {out}/06b_validation_m1.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 7: Wind Effects Demonstration
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 7: Wind Effects")

    wind_cases = [
        ("No Wind", 0, 0),
        ("Headwind 10 m/s", -10, 0),
        ("Tailwind 10 m/s", 10, 0),
        ("Crosswind 10 m/s", 0, 10),
    ]

    fig_wind, ax_wind = plt.subplots(figsize=(12, 6))
    fig_wind.patch.set_facecolor('#0a0a0a')
    ax_wind.set_facecolor('#0a0a0a')
    colors = ['#00d4ff', '#ff6b35', '#00e676', '#e040fb']

    for (label, wx, wz), color in zip(wind_cases, colors):
        cond_w = LaunchConditions(
            velocity=684.0, elevation_deg=45.0,
            wind_x=wx, wind_z=wz, latitude_deg=30.0,
        )
        r = simulate_rk4(proj, cond_w, dt=0.05, enable_coriolis=False)
        ax_wind.plot(r.x/1000, r.y/1000, color=color, linewidth=2, label=label)
        print(f"  {label:<25s}  Range: {r.range_total/1000:.2f} km")

    ax_wind.set_xlabel('Range (km)', color='#e0e0e0')
    ax_wind.set_ylabel('Altitude (km)', color='#e0e0e0')
    ax_wind.set_title('Effect of Wind on Trajectory', fontweight='bold', color='#e0e0e0')
    ax_wind.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='#e0e0e0')
    ax_wind.grid(True, color='#333', alpha=0.4)
    ax_wind.tick_params(colors='#e0e0e0')
    ax_wind.set_ylim(bottom=0)
    for s in ax_wind.spines.values():
        s.set_color('#333')
    fig_wind.tight_layout()
    fig_wind.savefig(f'{out}/07_wind_effects.png', dpi=150, facecolor='#0a0a0a',
                     bbox_inches='tight')
    plt.close(fig_wind)
    print(f"\n  ✓ Saved: {out}/07_wind_effects.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 8: Coriolis Effect Demonstration
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 8: Coriolis Effect")

    cond_long = LaunchConditions(
        velocity=684.0, elevation_deg=45.0,
        latitude_deg=30.0, azimuth_deg=0.0,
    )
    r_no_cor = simulate_rk4(proj, cond_long, dt=0.05, enable_coriolis=False)
    r_cor = simulate_rk4(proj, cond_long, dt=0.05, enable_coriolis=True)

    print(f"  Without Coriolis — Range: {r_no_cor.range_total/1000:.2f} km  "
          f"Drift: {r_no_cor.lateral_drift:.1f} m")
    print(f"  With Coriolis    — Range: {r_cor.range_total/1000:.2f} km  "
          f"Drift: {r_cor.lateral_drift:.1f} m")
    print(f"  Coriolis-induced drift: {r_cor.lateral_drift - r_no_cor.lateral_drift:.1f} m")

    fig_cor, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(14, 5))
    fig_cor.patch.set_facecolor('#0a0a0a')
    for ax in [ax_c1, ax_c2]:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#e0e0e0')
        ax.grid(True, color='#333', alpha=0.4)
        for s in ax.spines.values():
            s.set_color('#333')

    ax_c1.plot(r_no_cor.x/1000, r_no_cor.y/1000, '--', color='#888', linewidth=2,
               label='No Coriolis')
    ax_c1.plot(r_cor.x/1000, r_cor.y/1000, color='#00d4ff', linewidth=2,
               label='With Coriolis')
    ax_c1.set_xlabel('Range (km)', color='#e0e0e0')
    ax_c1.set_ylabel('Altitude (km)', color='#e0e0e0')
    ax_c1.set_title('Trajectory Side View', color='#e0e0e0', fontweight='bold')
    ax_c1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='#e0e0e0')
    ax_c1.set_ylim(bottom=0)

    ax_c2.plot(r_cor.x/1000, r_cor.z, color='#e040fb', linewidth=2)
    ax_c2.set_xlabel('Range (km)', color='#e0e0e0')
    ax_c2.set_ylabel('Lateral Drift (m)', color='#e0e0e0')
    ax_c2.set_title('Coriolis Drift (Top View)', color='#e0e0e0', fontweight='bold')
    ax_c2.axhline(y=0, color='#555', linestyle='--', alpha=0.5)

    fig_cor.suptitle('Coriolis Effect at 30°N Latitude',
                     fontsize=14, fontweight='bold', color='#e0e0e0', y=1.02)
    fig_cor.tight_layout()
    fig_cor.savefig(f'{out}/08_coriolis_effect.png', dpi=150, facecolor='#0a0a0a',
                    bbox_inches='tight')
    plt.close(fig_cor)
    print(f"\n  ✓ Saved: {out}/08_coriolis_effect.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 9: Full Dashboard
    # ══════════════════════════════════════════════════════════════════════
    section("PHASE 9: Full Dashboard")
    fig_dash = plot_dashboard(result_rk4, save_path=f'{out}/09_dashboard.png')
    plt.close(fig_dash)
    print(f"  ✓ Saved: {out}/09_dashboard.png")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 10: Trajectory Animation
    # ══════════════════════════════════════════════════════════════════════
    if not quick:
        section("PHASE 10: Trajectory Animation (GIF)")
        create_trajectory_animation(result_rk4,
                                    save_path=f'{out}/10_trajectory_animation.gif',
                                    frames=120)
        print(f"  ✓ Saved: {out}/10_trajectory_animation.gif")
    else:
        section("PHASE 10: Animation SKIPPED (--quick mode)")

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    section("COMPLETE")
    print(f"""
  All outputs saved to: {os.path.abspath(out)}/

  Generated files:
    01_atmosphere_profile.png    — ISA model (T, P, ρ, a vs altitude)
    02_cd_vs_mach.png            — Drag curves for all 4 geometries
    03_reference_trajectory.png  — Single trajectory plot
    04_geometry_comparison.png   — 4-panel geometry comparison
    05_euler_vs_rk4.png          — Numerical method comparison
    06_validation_m107.png       — 155mm M107 validation vs McCoy
    06b_validation_m1.png        — 105mm M1 validation vs McCoy
    07_wind_effects.png          — Wind effect demonstration
    08_coriolis_effect.png       — Earth rotation effect
    09_dashboard.png             — Full flight data dashboard
    {'10_trajectory_animation.gif — Animated trajectory' if not quick else '(animation skipped)'}

  Total runtime: {elapsed:.1f} seconds
""")


if __name__ == "__main__":
    main()
