"""
Visualization Engine
====================
Publication-quality plots for trajectory analysis:
  1. Trajectory (altitude vs range)
  2. Geometry comparison (all 4 shapes side-by-side)
  3. Velocity vs time
  4. Mach number vs time
  5. Cd vs Mach curves
  6. Atmospheric profile
  7. Dashboard with key metrics
  8. Validation comparison plots
  9. Euler vs RK4 accuracy comparison
  10. Animated trajectory (saved as GIF)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import List, Optional, Dict
import os

from .integrator import TrajectoryResult
from .drag_model import ALL_GEOMETRIES, DragModel
from .atmosphere import isa_profile


# ── Style Configuration ───────────────────────────────────────────────────
STYLE = {
    'bg_color': '#0a0a0a',
    'text_color': '#e0e0e0',
    'grid_color': '#333333',
    'accent_colors': ['#00d4ff', '#ff6b35', '#00e676', '#ffeb3b',
                      '#e040fb', '#ff5252'],
    'font_family': 'monospace',
}

def _apply_dark_style(fig, axes):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(STYLE['bg_color'])
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor(STYLE['bg_color'])
        ax.tick_params(colors=STYLE['text_color'])
        ax.xaxis.label.set_color(STYLE['text_color'])
        ax.yaxis.label.set_color(STYLE['text_color'])
        ax.title.set_color(STYLE['text_color'])
        ax.grid(True, color=STYLE['grid_color'], alpha=0.4, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(STYLE['grid_color'])


def ensure_output_dir(path: str = 'outputs'):
    os.makedirs(path, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  1. Single Trajectory Plot
# ══════════════════════════════════════════════════════════════════════════

def plot_trajectory(result: TrajectoryResult, save_path: str = None,
                    show: bool = False) -> plt.Figure:
    """Altitude vs downrange for a single trajectory."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_style(fig, ax)

    ax.plot(result.x / 1000, result.y / 1000,
            color=STYLE['accent_colors'][0], linewidth=2.5,
            label=f'{result.projectile.drag_model.name}')

    # Mark launch and impact
    ax.plot(0, result.y[0] / 1000, 'o', color='#00e676', markersize=10,
            label='Launch', zorder=5)
    ax.plot(result.x[-1] / 1000, result.y[-1] / 1000, 'x',
            color='#ff5252', markersize=12, markeredgewidth=3,
            label='Impact', zorder=5)

    # Mark max altitude
    idx_max = np.argmax(result.y)
    ax.plot(result.x[idx_max] / 1000, result.y[idx_max] / 1000, '^',
            color='#ffeb3b', markersize=10, label='Apex', zorder=5)

    ax.set_xlabel('Downrange (km)', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title(f'Projectile Trajectory — {result.projectile.name} '
                 f'({result.method.upper()}, v₀={result.conditions.velocity:.0f} m/s, '
                 f'θ={result.conditions.elevation_deg:.0f}°)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10,
              facecolor='#1a1a1a', edgecolor='#444', labelcolor=STYLE['text_color'])
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    if show:
        plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  2. Geometry Comparison
# ══════════════════════════════════════════════════════════════════════════

def plot_geometry_comparison(results: Dict[str, TrajectoryResult],
                            save_path: str = None) -> plt.Figure:
    """Side-by-side trajectories for different projectile geometries."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    _apply_dark_style(fig, axes)

    # Top-left: Trajectory comparison
    ax = axes[0, 0]
    for key, res in results.items():
        data = ALL_GEOMETRIES.get(key, {})
        color = data.get('color', '#ffffff')
        ax.plot(res.x / 1000, res.y / 1000, color=color, linewidth=2,
                label=f"{res.projectile.drag_model.name}")
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory Comparison', fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444',
              labelcolor=STYLE['text_color'])
    ax.set_ylim(bottom=0)

    # Top-right: Velocity vs time
    ax = axes[0, 1]
    for key, res in results.items():
        data = ALL_GEOMETRIES.get(key, {})
        color = data.get('color', '#ffffff')
        ax.plot(res.time, res.speed, color=color, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Velocity vs Time', fontweight='bold')

    # Bottom-left: Mach vs time
    ax = axes[1, 0]
    for key, res in results.items():
        data = ALL_GEOMETRIES.get(key, {})
        color = data.get('color', '#ffffff')
        ax.plot(res.time, res.mach_history, color=color, linewidth=2)
    ax.axhline(y=1.0, color='#ff5252', linestyle='--', alpha=0.6, label='Mach 1')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mach Number')
    ax.set_title('Mach Number vs Time', fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444',
              labelcolor=STYLE['text_color'])

    # Bottom-right: Bar chart of key metrics
    ax = axes[1, 1]
    names = [results[k].projectile.drag_model.name for k in results]
    ranges = [results[k].range_total / 1000 for k in results]
    colors = [ALL_GEOMETRIES.get(k, {}).get('color', '#888') for k in results]
    bars = ax.barh(names, ranges, color=colors, alpha=0.85, edgecolor='#555')
    ax.set_xlabel('Range (km)')
    ax.set_title('Range Comparison', fontweight='bold')
    for bar, r in zip(bars, ranges):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{r:.1f} km', va='center', color=STYLE['text_color'], fontsize=10)

    fig.suptitle('Projectile Geometry Comparison — Same Launch Conditions',
                 fontsize=15, fontweight='bold', color=STYLE['text_color'], y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  3. Cd vs Mach Curves
# ══════════════════════════════════════════════════════════════════════════

def plot_cd_vs_mach(save_path: str = None) -> plt.Figure:
    """Plot Cd vs Mach for all geometries."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _apply_dark_style(fig, ax)

    mach_range = np.linspace(0, 4.0, 500)
    for key, data in ALL_GEOMETRIES.items():
        model = DragModel(key)
        cd_vals = model.cd_array(mach_range)
        ax.plot(mach_range, cd_vals, color=data['color'],
                linestyle=data['linestyle'], linewidth=2.5,
                label=data['name'])

    # Annotate transonic region
    ax.axvspan(0.8, 1.2, alpha=0.08, color='#ff5252')
    ax.text(1.0, 0.05, 'Transonic\nRegion', ha='center',
            color='#ff5252', fontsize=10, alpha=0.7)

    ax.set_xlabel('Mach Number', fontsize=12)
    ax.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
    ax.set_title('Drag Coefficient vs Mach Number — Standard Geometries',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#444',
              labelcolor=STYLE['text_color'])
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  4. Atmospheric Profile
# ══════════════════════════════════════════════════════════════════════════

def plot_atmosphere(save_path: str = None) -> plt.Figure:
    """ISA atmospheric profile from 0 to 30 km."""
    altitudes = np.linspace(0, 30000, 500)
    profile = isa_profile(altitudes)

    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
    _apply_dark_style(fig, axes)

    alt_km = altitudes / 1000

    params = [
        ('Temperature (K)', profile['temperature'], '#ff6b35'),
        ('Pressure (Pa)', profile['pressure'], '#00d4ff'),
        ('Density (kg/m³)', profile['density'], '#00e676'),
        ('Speed of Sound (m/s)', profile['speed_of_sound'], '#ffeb3b'),
    ]

    for ax, (title, data, color) in zip(axes, params):
        ax.plot(data, alt_km, color=color, linewidth=2)
        ax.set_xlabel(title, fontsize=10)
        ax.fill_betweenx(alt_km, 0, data, alpha=0.1, color=color)

    axes[0].set_ylabel('Altitude (km)', fontsize=12)
    fig.suptitle('International Standard Atmosphere (ISA) Model',
                 fontsize=14, fontweight='bold', color=STYLE['text_color'])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  5. Dashboard
# ══════════════════════════════════════════════════════════════════════════

def plot_dashboard(result: TrajectoryResult, save_path: str = None) -> plt.Figure:
    """Comprehensive dashboard with trajectory + metrics + plots."""
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(STYLE['bg_color'])

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Trajectory (top, spans 2 cols) ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(STYLE['bg_color'])
    ax1.plot(result.x / 1000, result.y / 1000, color='#00d4ff', linewidth=2.5)
    idx_max = np.argmax(result.y)
    ax1.plot(result.x[idx_max]/1000, result.y[idx_max]/1000, '^',
             color='#ffeb3b', markersize=12)
    ax1.plot(result.x[-1]/1000, 0, 'x', color='#ff5252',
             markersize=14, markeredgewidth=3)
    ax1.set_xlabel('Range (km)', color=STYLE['text_color'])
    ax1.set_ylabel('Altitude (km)', color=STYLE['text_color'])
    ax1.set_title('TRAJECTORY', fontweight='bold', color=STYLE['text_color'], fontsize=13)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, color=STYLE['grid_color'], alpha=0.3)
    ax1.tick_params(colors=STYLE['text_color'])
    for spine in ax1.spines.values():
        spine.set_color(STYLE['grid_color'])

    # ── Metrics panel (top-right) ──
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.set_facecolor('#111111')
    ax_info.axis('off')

    metrics = [
        ('LAUNCH', f'{result.conditions.velocity:.0f} m/s @ {result.conditions.elevation_deg:.0f}°'),
        ('RANGE', f'{result.range_total/1000:.2f} km'),
        ('MAX ALT', f'{result.max_altitude/1000:.2f} km'),
        ('FLIGHT TIME', f'{result.flight_time:.1f} s'),
        ('IMPACT VEL', f'{result.impact_velocity:.0f} m/s'),
        ('IMPACT ANGLE', f'{result.impact_angle_deg:.1f}°'),
        ('DRIFT', f'{result.lateral_drift:.1f} m'),
        ('METHOD', result.method.upper()),
    ]

    for i, (label, value) in enumerate(metrics):
        y_pos = 0.92 - i * 0.115
        ax_info.text(0.05, y_pos, label, fontsize=10, fontweight='bold',
                     color='#888888', transform=ax_info.transAxes, fontfamily='monospace')
        ax_info.text(0.95, y_pos, value, fontsize=11, fontweight='bold',
                     color='#00d4ff', transform=ax_info.transAxes,
                     ha='right', fontfamily='monospace')

    ax_info.set_title('FLIGHT DATA', fontweight='bold',
                      color=STYLE['text_color'], fontsize=13, pad=10)

    # ── Velocity vs time (middle-left) ──
    ax2 = fig.add_subplot(gs[1, 0])
    _apply_dark_style(fig, ax2)
    ax2.plot(result.time, result.speed, color='#ff6b35', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('SPEED', fontweight='bold')

    # ── Mach vs time (middle-center) ──
    ax3 = fig.add_subplot(gs[1, 1])
    _apply_dark_style(fig, ax3)
    ax3.plot(result.time, result.mach_history, color='#e040fb', linewidth=2)
    ax3.axhline(y=1.0, color='#ff5252', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mach')
    ax3.set_title('MACH NUMBER', fontweight='bold')

    # ── Cd vs time (middle-right) ──
    ax4 = fig.add_subplot(gs[1, 2])
    _apply_dark_style(fig, ax4)
    ax4.plot(result.time, result.cd_history, color='#00e676', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Cd')
    ax4.set_title('DRAG COEFFICIENT', fontweight='bold')

    # ── Altitude vs time (bottom-left) ──
    ax5 = fig.add_subplot(gs[2, 0])
    _apply_dark_style(fig, ax5)
    ax5.plot(result.time, result.y / 1000, color='#ffeb3b', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Altitude (km)')
    ax5.set_title('ALTITUDE', fontweight='bold')

    # ── Air density vs time (bottom-center) ──
    ax6 = fig.add_subplot(gs[2, 1])
    _apply_dark_style(fig, ax6)
    ax6.plot(result.time, result.density_history, color='#26c6da', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('ρ (kg/m³)')
    ax6.set_title('AIR DENSITY', fontweight='bold')

    # ── Velocity components (bottom-right) ──
    ax7 = fig.add_subplot(gs[2, 2])
    _apply_dark_style(fig, ax7)
    ax7.plot(result.time, result.vx, label='vx (range)', color='#00d4ff', linewidth=1.5)
    ax7.plot(result.time, result.vy, label='vy (vertical)', color='#ff6b35', linewidth=1.5)
    if np.max(np.abs(result.vz)) > 0.1:
        ax7.plot(result.time, result.vz, label='vz (lateral)', color='#00e676', linewidth=1.5)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Velocity (m/s)')
    ax7.set_title('VELOCITY COMPONENTS', fontweight='bold')
    ax7.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444',
               labelcolor=STYLE['text_color'])

    fig.suptitle(f'PROJECTILE DYNAMICS DASHBOARD — {result.projectile.name}',
                 fontsize=16, fontweight='bold', color='#00d4ff', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  6. Euler vs RK4 Comparison
# ══════════════════════════════════════════════════════════════════════════

def plot_euler_vs_rk4(euler_result: TrajectoryResult,
                      rk4_result: TrajectoryResult,
                      save_path: str = None) -> plt.Figure:
    """Compare Euler and RK4 trajectories to show accuracy difference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _apply_dark_style(fig, axes)

    # Trajectory
    ax = axes[0]
    ax.plot(euler_result.x/1000, euler_result.y/1000,
            color='#ff6b35', linewidth=2, linestyle='--', label=f'Euler (dt={euler_result.dt}s)')
    ax.plot(rk4_result.x/1000, rk4_result.y/1000,
            color='#00d4ff', linewidth=2, label=f'RK4 (dt={rk4_result.dt}s)')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Trajectory Comparison', fontweight='bold')
    ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#444',
              labelcolor=STYLE['text_color'])
    ax.set_ylim(bottom=0)

    # Speed
    ax = axes[1]
    ax.plot(euler_result.time, euler_result.speed,
            color='#ff6b35', linewidth=2, linestyle='--', label='Euler')
    ax.plot(rk4_result.time, rk4_result.speed,
            color='#00d4ff', linewidth=2, label='RK4')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed vs Time', fontweight='bold')
    ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#444',
              labelcolor=STYLE['text_color'])

    # Metrics comparison
    ax = axes[2]
    ax.axis('off')
    ax.set_facecolor('#111111')

    text_lines = [
        f"{'Metric':<18} {'Euler':>12} {'RK4':>12} {'Δ':>10}",
        f"{'─'*52}",
        f"{'Range (km)':<18} {euler_result.range_total/1000:>12.2f} "
        f"{rk4_result.range_total/1000:>12.2f} "
        f"{(euler_result.range_total - rk4_result.range_total)/1000:>+10.2f}",
        f"{'Max Alt (km)':<18} {euler_result.max_altitude/1000:>12.2f} "
        f"{rk4_result.max_altitude/1000:>12.2f} "
        f"{(euler_result.max_altitude - rk4_result.max_altitude)/1000:>+10.2f}",
        f"{'Flight Time (s)':<18} {euler_result.flight_time:>12.2f} "
        f"{rk4_result.flight_time:>12.2f} "
        f"{euler_result.flight_time - rk4_result.flight_time:>+10.2f}",
        f"{'Impact Vel (m/s)':<18} {euler_result.impact_velocity:>12.1f} "
        f"{rk4_result.impact_velocity:>12.1f} "
        f"{euler_result.impact_velocity - rk4_result.impact_velocity:>+10.1f}",
    ]

    ax.text(0.05, 0.85, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', color=STYLE['text_color'],
            verticalalignment='top')
    ax.set_title('Numerical Comparison', fontweight='bold',
                 color=STYLE['text_color'])

    fig.suptitle('Euler vs Runge-Kutta 4th Order — Accuracy Comparison',
                 fontsize=14, fontweight='bold', color=STYLE['text_color'], y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  7. Validation Plot
# ══════════════════════════════════════════════════════════════════════════

def plot_validation(validation_results, reference_data: dict,
                    save_path: str = None) -> plt.Figure:
    """Plot simulated vs reference range for validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_style(fig, axes)

    elevations = [v.elevation_deg for v in validation_results]
    ref_ranges = [v.ref_range / 1000 for v in validation_results]
    sim_ranges = [v.sim_range / 1000 for v in validation_results]
    errors = [v.range_error_pct for v in validation_results]

    # Range comparison
    ax = axes[0]
    ax.plot(elevations, ref_ranges, 'o-', color='#ffeb3b', linewidth=2,
            markersize=8, label='Reference (McCoy)')
    ax.plot(elevations, sim_ranges, 's--', color='#00d4ff', linewidth=2,
            markersize=8, label='Simulation (RK4)')
    ax.set_xlabel('Elevation Angle (°)')
    ax.set_ylabel('Range (km)')
    ax.set_title(f'Range Validation — {reference_data["name"]}', fontweight='bold')
    ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#444',
              labelcolor=STYLE['text_color'])

    # Error bars
    ax = axes[1]
    colors = ['#00e676' if abs(e) < 10 else '#ff5252' for e in errors]
    ax.bar(elevations, errors, color=colors, alpha=0.8, width=2)
    ax.axhline(y=0, color='#888', linewidth=0.5)
    ax.axhspan(-10, 10, alpha=0.05, color='#00e676')
    ax.set_xlabel('Elevation Angle (°)')
    ax.set_ylabel('Range Error (%)')
    ax.set_title('Validation Error', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg_color'])
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  8. Animated Trajectory (GIF)
# ══════════════════════════════════════════════════════════════════════════

def create_trajectory_animation(result: TrajectoryResult,
                                save_path: str = 'outputs/trajectory_anim.gif',
                                frames: int = 100) -> str:
    """Create animated GIF of trajectory with trail."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(STYLE['bg_color'])
    ax.set_facecolor(STYLE['bg_color'])

    x_km = result.x / 1000
    y_km = result.y / 1000

    ax.set_xlim(0, max(x_km) * 1.05)
    ax.set_ylim(0, max(y_km) * 1.15)
    ax.set_xlabel('Range (km)', color=STYLE['text_color'], fontsize=12)
    ax.set_ylabel('Altitude (km)', color=STYLE['text_color'], fontsize=12)
    ax.set_title(f'Trajectory Animation — {result.projectile.name}',
                 color=STYLE['text_color'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=STYLE['text_color'])
    ax.grid(True, color=STYLE['grid_color'], alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(STYLE['grid_color'])

    trail_line, = ax.plot([], [], color='#00d4ff', linewidth=1.5, alpha=0.6)
    point, = ax.plot([], [], 'o', color='#00d4ff', markersize=8)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        color=STYLE['text_color'], fontsize=11, fontfamily='monospace')

    # Subsample for animation
    total_pts = len(x_km)
    step = max(1, total_pts // frames)
    indices = list(range(0, total_pts, step))
    if indices[-1] != total_pts - 1:
        indices.append(total_pts - 1)

    def animate(frame_idx):
        idx = indices[min(frame_idx, len(indices) - 1)]
        trail_line.set_data(x_km[:idx+1], y_km[:idx+1])
        point.set_data([x_km[idx]], [y_km[idx]])
        t = result.time[idx]
        spd = result.speed[idx]
        alt = result.y[idx]
        time_text.set_text(
            f't={t:.1f}s | v={spd:.0f} m/s | alt={alt:.0f} m | '
            f'Mach={result.mach_history[idx]:.2f}'
        )
        return trail_line, point, time_text

    anim = FuncAnimation(fig, animate, frames=len(indices), interval=50, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=20),
              savefig_kwargs={'facecolor': STYLE['bg_color']})
    plt.close(fig)
    print(f"  Animation saved: {save_path}")
    return save_path
