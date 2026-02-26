"""
Unit Tests for Projectile Dynamics Simulator
=============================================
Tests core physics modules for correctness.
Run: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.atmosphere import (
    isa_temperature, isa_pressure, isa_density,
    speed_of_sound, SEA_LEVEL_TEMP, SEA_LEVEL_PRESSURE, SEA_LEVEL_DENSITY,
)
from src.drag_model import DragModel, drag_force, ALL_GEOMETRIES
from src.projectile import Projectile, LaunchConditions, compute_forces
from src.integrator import simulate_euler, simulate_rk4


class TestAtmosphere:
    """Verify ISA model against known standard values."""

    def test_sea_level_temperature(self):
        assert abs(isa_temperature(0) - 288.15) < 0.01

    def test_sea_level_pressure(self):
        assert abs(isa_pressure(0) - 101325.0) < 1.0

    def test_sea_level_density(self):
        assert abs(isa_density(0) - 1.225) < 0.01

    def test_tropopause_temperature(self):
        """Temperature at 11 km should be ~216.65 K."""
        assert abs(isa_temperature(11000) - 216.65) < 0.5

    def test_density_decreases_with_altitude(self):
        """Air density must decrease monotonically."""
        rho_0 = isa_density(0)
        rho_5 = isa_density(5000)
        rho_10 = isa_density(10000)
        assert rho_0 > rho_5 > rho_10

    def test_speed_of_sound_sea_level(self):
        """Speed of sound at sea level ~340.3 m/s."""
        a = speed_of_sound(0)
        assert abs(a - 340.3) < 1.0

    def test_pressure_decreases_with_altitude(self):
        assert isa_pressure(5000) < isa_pressure(0)
        assert isa_pressure(10000) < isa_pressure(5000)


class TestDragModel:
    """Verify drag coefficient interpolation."""

    def test_all_geometries_exist(self):
        for key in ['flat_nose', 'sphere', 'ogive', 'ogive_boattail']:
            model = DragModel(key)
            assert model.name is not None

    def test_cd_positive(self):
        for key in ALL_GEOMETRIES:
            model = DragModel(key)
            for mach in [0.0, 0.5, 1.0, 2.0, 3.0]:
                assert model.cd(mach) > 0

    def test_transonic_drag_rise(self):
        """Cd should spike in the transonic region (Mach 0.8-1.2)."""
        for key in ALL_GEOMETRIES:
            model = DragModel(key)
            cd_sub = model.cd(0.5)
            cd_trans = model.cd(1.0)
            assert cd_trans > cd_sub

    def test_ogive_boattail_lowest_drag(self):
        """Ogive+boat-tail should have lowest drag at all Mach numbers."""
        obt = DragModel('ogive_boattail')
        for key in ['flat_nose', 'sphere']:
            other = DragModel(key)
            for mach in [0.5, 1.0, 2.0]:
                assert obt.cd(mach) < other.cd(mach)

    def test_drag_force_opposes_motion(self):
        """Drag force must oppose velocity direction."""
        v = np.array([100.0, 50.0, 0.0])
        F = drag_force(v, rho=1.225, cd=0.3, area=0.01)
        # Dot product should be negative (opposing)
        assert np.dot(F, v) < 0

    def test_drag_force_zero_at_rest(self):
        F = drag_force(np.array([0.0, 0.0, 0.0]), rho=1.225, cd=0.3, area=0.01)
        assert np.allclose(F, 0.0)


class TestProjectile:
    """Verify projectile and force computation."""

    def test_projectile_area(self):
        p = Projectile(diameter=0.155)
        expected = np.pi * (0.155 / 2) ** 2
        assert abs(p.area - expected) < 1e-8

    def test_initial_velocity_vector(self):
        cond = LaunchConditions(velocity=100.0, elevation_deg=45.0, azimuth_deg=0.0)
        v = cond.initial_velocity_vector()
        assert abs(np.linalg.norm(v) - 100.0) < 0.01
        assert abs(v[1] - 100 * np.sin(np.radians(45))) < 0.01

    def test_gravity_dominates_at_low_speed(self):
        """At very low speed, gravity should dominate acceleration."""
        proj = Projectile(mass=10.0, diameter=0.05, geometry='sphere')
        cond = LaunchConditions(velocity=1.0, elevation_deg=0.0)
        pos = np.array([0.0, 100.0, 0.0])
        vel = np.array([1.0, 0.0, 0.0])
        acc = compute_forces(pos, vel, proj, cond, enable_coriolis=False)
        assert abs(acc[1] - (-9.80665)) < 0.5  # mostly gravity


class TestIntegrators:
    """Verify numerical integration methods."""

    def test_euler_runs(self):
        proj = Projectile(mass=10.0, diameter=0.1, geometry='sphere')
        cond = LaunchConditions(velocity=300.0, elevation_deg=45.0)
        result = simulate_euler(proj, cond, dt=0.1, enable_coriolis=False)
        assert result.flight_time > 0
        assert result.range_total > 0

    def test_rk4_runs(self):
        proj = Projectile(mass=10.0, diameter=0.1, geometry='sphere')
        cond = LaunchConditions(velocity=300.0, elevation_deg=45.0)
        result = simulate_rk4(proj, cond, dt=0.1, enable_coriolis=False)
        assert result.flight_time > 0
        assert result.range_total > 0

    def test_rk4_more_accurate_than_euler(self):
        """At same large timestep, RK4 should be closer to fine-dt reference."""
        proj = Projectile(mass=10.0, diameter=0.1, geometry='ogive')
        cond = LaunchConditions(velocity=500.0, elevation_deg=30.0)

        # Reference: RK4 with very small timestep
        ref = simulate_rk4(proj, cond, dt=0.001, enable_coriolis=False)

        # Compare: large timestep
        euler = simulate_euler(proj, cond, dt=0.5, enable_coriolis=False)
        rk4 = simulate_rk4(proj, cond, dt=0.5, enable_coriolis=False)

        euler_err = abs(euler.range_total - ref.range_total)
        rk4_err = abs(rk4.range_total - ref.range_total)
        assert rk4_err < euler_err

    def test_projectile_hits_ground(self):
        """Trajectory should end near ground level."""
        proj = Projectile(mass=10.0, diameter=0.1, geometry='ogive')
        cond = LaunchConditions(velocity=500.0, elevation_deg=45.0)
        result = simulate_rk4(proj, cond, dt=0.05)
        assert result.y[-1] <= 0.5  # close to ground

    def test_headwind_reduces_range(self):
        """Headwind should reduce range."""
        proj = Projectile(mass=10.0, diameter=0.1, geometry='ogive')
        cond_no_wind = LaunchConditions(velocity=500.0, elevation_deg=30.0)
        cond_headwind = LaunchConditions(velocity=500.0, elevation_deg=30.0, wind_x=-20.0)

        r1 = simulate_rk4(proj, cond_no_wind, dt=0.05, enable_coriolis=False)
        r2 = simulate_rk4(proj, cond_headwind, dt=0.05, enable_coriolis=False)
        assert r2.range_total < r1.range_total

    def test_higher_drag_reduces_range(self):
        """Flat nose (high drag) should have less range than ogive+bt."""
        cond = LaunchConditions(velocity=500.0, elevation_deg=45.0)
        p_flat = Projectile(mass=10.0, diameter=0.1, geometry='flat_nose')
        p_obt = Projectile(mass=10.0, diameter=0.1, geometry='ogive_boattail')

        r_flat = simulate_rk4(p_flat, cond, dt=0.05, enable_coriolis=False)
        r_obt = simulate_rk4(p_obt, cond, dt=0.05, enable_coriolis=False)
        assert r_obt.range_total > r_flat.range_total


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
