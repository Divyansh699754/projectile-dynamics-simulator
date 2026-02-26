# Physics Notes — Projectile Dynamics Simulator

## 1. Equations of Motion

The state of the projectile at any time is described by position **r** = [x, y, z] and velocity **v** = [vx, vy, vz].

The equations of motion are:

```
dr/dt = v
dv/dt = a_gravity + a_drag + a_coriolis
```

These are coupled, nonlinear ODEs that cannot be solved analytically when drag depends on velocity.

## 2. Gravity

Simple constant acceleration:
```
a_gravity = [0, -g, 0]
g = 9.80665 m/s²
```

## 3. Aerodynamic Drag

The drag force is:
```
F_drag = -½ × ρ(h) × |v_rel|² × Cd(M) × A × v̂_rel
```

Where:
- **ρ(h)** — air density at altitude h (from ISA model)
- **|v_rel|** — speed relative to air mass (accounts for wind)
- **Cd(M)** — drag coefficient as function of Mach number
- **A** — reference cross-sectional area (π × d²/4)
- **v̂_rel** — unit vector in direction of relative velocity

The negative sign means drag always opposes motion.

### Why Cd varies with Mach

At subsonic speeds, drag is primarily from skin friction and form drag. As the projectile approaches Mach 1 (transonic), shock waves form on the body surface, dramatically increasing wave drag. This creates the characteristic "transonic drag rise." At supersonic speeds, the shock structure stabilizes and Cd decreases gradually.

## 4. International Standard Atmosphere (ISA 1976)

### Troposphere (0–11 km)
```
T(h) = T₀ + L × h
P(h) = P₀ × (T/T₀)^(g×M / R×|L|)
ρ(h) = P(h) / (R_specific × T(h))
```
Where L = -0.0065 K/m (lapse rate).

### Lower Stratosphere (11–20 km)
Temperature is constant at 216.65 K. Pressure decays exponentially:
```
P(h) = P_tropo × exp(-g×M×(h - h_tropo) / (R × T_tropo))
```

### Speed of Sound
```
a = √(γ × R_specific × T)
```
Where γ = 1.4 for dry air.

## 5. Wind Effects

Wind is modeled as a constant vector [wx, wy, wz] representing the air mass velocity. The aerodynamic forces are computed using velocity **relative to the air**:

```
v_rel = v_projectile - v_wind
```

This means:
- Headwind (wx > 0) → increases relative speed → more drag → shorter range
- Tailwind (wx < 0) → decreases relative speed → less drag → longer range
- Crosswind (wz ≠ 0) → lateral drag component → projectile drifts

## 6. Coriolis Effect

Earth rotates at Ω = 7.2921 × 10⁻⁵ rad/s. The Coriolis acceleration in the local frame is:

```
a_coriolis = -2(Ω × v)
```

Where Ω in the local frame (x=North, y=Up, z=East):
```
Ω = Ω_earth × [cos(φ), sin(φ), 0]
```
φ = latitude

For a projectile fired northward at 30°N latitude, the Coriolis effect deflects it to the right (east) by tens of meters over a 15 km trajectory.

## 7. Numerical Methods

### Euler Method
```
x_{n+1} = x_n + v_n × dt
v_{n+1} = v_n + a(x_n, v_n) × dt
```
Error: O(dt) per step → O(dt) global error. Simple but inaccurate.

### 4th-Order Runge-Kutta (RK4)
Evaluates the derivative at four points per step:
```
k1 = f(t_n, y_n)
k2 = f(t_n + dt/2, y_n + dt/2 × k1)
k3 = f(t_n + dt/2, y_n + dt/2 × k2)
k4 = f(t_n + dt, y_n + dt × k3)

y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
```
Error: O(dt⁴) per step → O(dt⁴) global error. Much better accuracy.

## 8. Projectile Geometry Effects

The shape of a projectile dramatically affects its drag profile:

- **Flat nose**: Massive flow separation behind the flat face. Highest drag.
- **Sphere**: Better than flat, but still significant pressure drag.
- **Ogive nose**: Pointed shape allows air to flow smoothly around it. Much lower drag.
- **Ogive + boat-tail**: Tapered rear reduces base drag (low-pressure wake). Lowest overall drag.

This is why virtually all long-range projectiles use an ogive + boat-tail design.
