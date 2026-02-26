"""
International Standard Atmosphere (ISA) Model
==============================================
Computes air density, pressure, temperature, and speed of sound
as functions of geometric altitude using the ISA 1976 model.

Valid from sea level to 86 km altitude.
Implements the troposphere (0-11 km) and lower stratosphere (11-20 km)
layers with their respective lapse rates.

Reference: U.S. Standard Atmosphere, 1976 (NASA-TM-X-74335)
"""

import numpy as np


# ── ISA Constants ──────────────────────────────────────────────────────────
SEA_LEVEL_TEMP       = 288.15      # K  (15 °C)
SEA_LEVEL_PRESSURE   = 101325.0    # Pa
SEA_LEVEL_DENSITY    = 1.225       # kg/m³
LAPSE_RATE_TROPO     = -0.0065     # K/m  (troposphere)
TROPOPAUSE_ALT       = 11000.0     # m
TROPOPAUSE_TEMP      = 216.65      # K  (-56.5 °C)
GRAVITY              = 9.80665     # m/s²
MOLAR_MASS_AIR       = 0.0289644   # kg/mol
GAS_CONSTANT         = 8.31447     # J/(mol·K)
SPECIFIC_HEAT_RATIO  = 1.4         # γ for dry air
R_SPECIFIC           = 287.058     # J/(kg·K)  specific gas constant for air


def isa_temperature(altitude: float) -> float:
    """
    Temperature at a given geometric altitude (m).

    - Troposphere (0–11 km): linear lapse at −6.5 °C/km
    - Stratosphere (11–20 km): isothermal at 216.65 K
    - Above 20 km: second stratosphere lapse +1 °C/km (simplified)
    """
    if altitude <= TROPOPAUSE_ALT:
        return SEA_LEVEL_TEMP + LAPSE_RATE_TROPO * altitude
    elif altitude <= 20000.0:
        return TROPOPAUSE_TEMP
    elif altitude <= 32000.0:
        return TROPOPAUSE_TEMP + 0.001 * (altitude - 20000.0)
    else:
        # Above 32 km — simplified extension
        return TROPOPAUSE_TEMP + 12.0 + 0.0028 * (altitude - 32000.0)


def isa_pressure(altitude: float) -> float:
    """
    Atmospheric pressure (Pa) at a given geometric altitude (m).
    Uses barometric formula appropriate for each layer.
    """
    exponent = GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * abs(LAPSE_RATE_TROPO))

    if altitude <= TROPOPAUSE_ALT:
        T = isa_temperature(altitude)
        return SEA_LEVEL_PRESSURE * (T / SEA_LEVEL_TEMP) ** exponent
    else:
        # Pressure at tropopause
        P_tropo = SEA_LEVEL_PRESSURE * (TROPOPAUSE_TEMP / SEA_LEVEL_TEMP) ** exponent
        if altitude <= 20000.0:
            # Isothermal layer — exponential decay
            return P_tropo * np.exp(
                -GRAVITY * MOLAR_MASS_AIR * (altitude - TROPOPAUSE_ALT)
                / (GAS_CONSTANT * TROPOPAUSE_TEMP)
            )
        else:
            # Simplified for upper layers
            P_20 = P_tropo * np.exp(
                -GRAVITY * MOLAR_MASS_AIR * (20000.0 - TROPOPAUSE_ALT)
                / (GAS_CONSTANT * TROPOPAUSE_TEMP)
            )
            T_20 = isa_temperature(20000.0)
            T_h = isa_temperature(altitude)
            lapse = 0.001  # K/m
            exp2 = GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * lapse)
            return P_20 * (T_h / T_20) ** (-exp2)


def isa_density(altitude: float) -> float:
    """
    Air density (kg/m³) from ideal gas law: ρ = P / (R_specific × T).
    """
    T = isa_temperature(altitude)
    P = isa_pressure(altitude)
    return P / (R_SPECIFIC * T)


def speed_of_sound(altitude: float) -> float:
    """
    Local speed of sound (m/s) = sqrt(γ × R_specific × T).
    """
    T = isa_temperature(altitude)
    return np.sqrt(SPECIFIC_HEAT_RATIO * R_SPECIFIC * T)


def mach_number(velocity_magnitude: float, altitude: float) -> float:
    """
    Mach number = |v| / a(h).
    """
    a = speed_of_sound(altitude)
    return velocity_magnitude / a if a > 0 else 0.0


# ── Vectorized versions for plotting ──────────────────────────────────────
def isa_profile(alt_array: np.ndarray) -> dict:
    """
    Compute full atmospheric profile for an array of altitudes.
    Returns dict with keys: 'temperature', 'pressure', 'density', 'speed_of_sound'.
    """
    T = np.array([isa_temperature(h) for h in alt_array])
    P = np.array([isa_pressure(h) for h in alt_array])
    rho = np.array([isa_density(h) for h in alt_array])
    a = np.array([speed_of_sound(h) for h in alt_array])
    return {
        'altitude': alt_array,
        'temperature': T,
        'pressure': P,
        'density': rho,
        'speed_of_sound': a,
    }


if __name__ == "__main__":
    # Quick validation print
    print("ISA Model Verification")
    print("=" * 60)
    print(f"{'Alt (m)':>10} {'T (K)':>10} {'P (Pa)':>12} {'ρ (kg/m³)':>12} {'a (m/s)':>10}")
    print("-" * 60)
    for h in [0, 1000, 5000, 10000, 11000, 15000, 20000]:
        T = isa_temperature(h)
        P = isa_pressure(h)
        rho = isa_density(h)
        a = speed_of_sound(h)
        print(f"{h:>10.0f} {T:>10.2f} {P:>12.1f} {rho:>12.5f} {a:>10.2f}")
