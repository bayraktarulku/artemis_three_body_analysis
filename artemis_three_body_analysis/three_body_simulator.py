"""
Artemis II Free-Return Trajectory Simulator
Restricted Three-Body Problem with Numerical Integration

This module implements a high-fidelity simulator for lunar free-return trajectories
similar to those used in the Artemis II mission. It uses the Circular Restricted
Three-Body Problem (CR3BP) formulation in Earth-Moon rotating frame.

Author: Trajectory Analysis Team
Date: February 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import json


@dataclass
class PhysicalConstants:
    """Physical constants for Earth-Moon system"""
    # Gravitational parameters (km³/s²)
    MU_EARTH = 398600.4418  # Earth's gravitational parameter
    MU_MOON = 4902.8000     # Moon's gravitational parameter

    # System properties
    R_EARTH_MOON = 384400.0  # Earth-Moon distance (km)

    # Mass ratio (μ = M_moon / (M_earth + M_moon))
    @property
    def mu(self) -> float:
        return self.MU_MOON / (self.MU_EARTH + self.MU_MOON)

    # Characteristic velocity (km/s)
    @property
    def v_char(self) -> float:
        return np.sqrt((self.MU_EARTH + self.MU_MOON) / self.R_EARTH_MOON)

    # Characteristic time (seconds)
    @property
    def t_char(self) -> float:
        return self.R_EARTH_MOON / self.v_char


class CR3BPDynamics:
    """
    Circular Restricted Three-Body Problem Dynamics

    Equations of motion in Earth-Moon rotating frame:
    ẍ - 2ẏ = ∂U/∂x
    ÿ + 2ẋ = ∂U/∂y
    z̈ = ∂U/∂z

    where U is the effective potential (pseudo-potential)
    """

    def __init__(self, constants: PhysicalConstants):
        self.const = constants
        self.mu = constants.mu

    def effective_potential(self, x: float, y: float, z: float) -> float:
        """
        Calculate effective potential U(x,y,z)
        U = (1-μ)/r₁ + μ/r₂ + ½(x² + y²)
        """
        # Distance to Earth (primary body at -μ, 0, 0)
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)

        # Distance to Moon (secondary body at 1-μ, 0, 0)
        r2 = np.sqrt((x - (1 - self.mu))**2 + y**2 + z**2)

        # Effective potential
        U = (1 - self.mu) / r1 + self.mu / r2 + 0.5 * (x**2 + y**2)

        return U

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        CR3BP equations of motion

        State vector: [x, y, z, vx, vy, vz]
        Returns: [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state

        # Distances to primaries
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu))**2 + y**2 + z**2)

        # Prevent singularities
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        # Gravitational accelerations
        ax = (x + 2*vy
              - (1 - self.mu) * (x + self.mu) / r1**3
              - self.mu * (x - (1 - self.mu)) / r2**3)

        ay = (y - 2*vx
              - (1 - self.mu) * y / r1**3
              - self.mu * y / r2**3)

        az = (-(1 - self.mu) * z / r1**3
              - self.mu * z / r2**3)

        return np.array([vx, vy, vz, ax, ay, az])

    def jacobi_constant(self, state: np.ndarray) -> float:
        """
        Calculate Jacobi constant (integral of motion)
        C = 2U - v²

        This should remain constant along the trajectory (energy-like invariant)
        """
        x, y, z, vx, vy, vz = state

        U = self.effective_potential(x, y, z)
        v_squared = vx**2 + vy**2 + vz**2

        return 2 * U - v_squared


class NumericalIntegrator:
    """
    Numerical integration schemes for trajectory propagation
    Includes error analysis and adaptive stepping
    """

    def __init__(self, dynamics: CR3BPDynamics):
        self.dynamics = dynamics

    def integrate_trajectory(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        method: str = 'DOP853',
        rtol: float = 1e-12,
        atol: float = 1e-12,
        max_step: float = 0.01
    ) -> Dict:
        """
        Integrate trajectory using high-order method

        Args:
            initial_state: [x, y, z, vx, vy, vz] in normalized units
            t_span: (t_start, t_end) in normalized time
            method: Integration method (DOP853, RK45, etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum step size

        Returns:
            Dictionary with solution and diagnostics
        """
        # Event detection for Earth/Moon close approaches
        def earth_proximity(t, y):
            r_earth = np.sqrt((y[0] + self.dynamics.mu)**2 + y[1]**2 + y[2]**2)
            return r_earth - 0.02  # 2% of Earth-Moon distance

        def moon_proximity(t, y):
            r_moon = np.sqrt((y[0] - (1 - self.dynamics.mu))**2 + y[1]**2 + y[2]**2)
            return r_moon - 0.01  # 1% of Earth-Moon distance

        earth_proximity.terminal = False
        moon_proximity.terminal = False

        # Integrate
        solution = solve_ivp(
            self.dynamics.equations_of_motion,
            t_span,
            initial_state,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
            events=[earth_proximity, moon_proximity]
        )

        # Calculate error metrics
        jacobi_values = np.array([
            self.dynamics.jacobi_constant(solution.y[:, i])
            for i in range(solution.y.shape[1])
        ])

        jacobi_initial = jacobi_values[0]
        jacobi_error = np.abs(jacobi_values - jacobi_initial)
        max_jacobi_error = np.max(jacobi_error)

        return {
            'solution': solution,
            'time': solution.t,
            'state': solution.y,
            'success': solution.success,
            'message': solution.message,
            'n_evaluations': solution.nfev,
            'jacobi_initial': jacobi_initial,
            'jacobi_error': jacobi_error,
            'max_jacobi_error': max_jacobi_error,
            'jacobi_relative_error': max_jacobi_error / abs(jacobi_initial) if jacobi_initial != 0 else 0
        }

    def error_analysis(self, result: Dict) -> Dict:
        """
        Comprehensive error analysis of integration
        """
        state = result['state']
        time = result['time']

        # Position and velocity magnitudes
        position = state[0:3, :]
        velocity = state[3:6, :]

        pos_magnitude = np.linalg.norm(position, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=0)

        # Energy-like metric (Jacobi constant drift)
        jacobi_drift = result['jacobi_error']

        return {
            'time_span': (time[0], time[-1]),
            'duration_days': (time[-1] - time[0]) * self.dynamics.const.t_char / 86400,
            'max_jacobi_drift': result['max_jacobi_error'],
            'relative_jacobi_drift': result['jacobi_relative_error'],
            'position_range': (np.min(pos_magnitude), np.max(pos_magnitude)),
            'velocity_range': (np.min(vel_magnitude), np.max(vel_magnitude)),
            'function_evaluations': result['n_evaluations'],
            'integration_success': result['success']
        }


class ArtemisIITrajectory:
    """
    Artemis II mission trajectory simulator
    Implements free-return lunar flyby trajectory
    """

    def __init__(self):
        self.constants = PhysicalConstants()
        self.dynamics = CR3BPDynamics(self.constants)
        self.integrator = NumericalIntegrator(self.dynamics)

    def dimensional_to_normalized(
        self,
        position_km: np.ndarray,
        velocity_km_s: np.ndarray
    ) -> np.ndarray:
        """Convert dimensional state to normalized CR3BP units"""
        # Position: normalize by Earth-Moon distance
        pos_norm = position_km / self.constants.R_EARTH_MOON

        # Velocity: normalize by characteristic velocity
        vel_norm = velocity_km_s / self.constants.v_char

        return np.concatenate([pos_norm, vel_norm])

    def normalized_to_dimensional(
        self,
        state_normalized: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert normalized state to dimensional units"""
        position_km = state_normalized[0:3] * self.constants.R_EARTH_MOON
        velocity_km_s = state_normalized[3:6] * self.constants.v_char

        return position_km, velocity_km_s

    def create_free_return_initial_conditions(
        self,
        altitude_km: float = 300.0,
        inclination_deg: float = 30.0,
        c3_energy: float = -0.5
    ) -> np.ndarray:
        """
        Generate initial conditions for free-return trajectory

        Args:
            altitude_km: Parking orbit altitude above Earth (km)
            inclination_deg: Orbital inclination (degrees)
            c3_energy: Characteristic energy (km²/s²)

        Returns:
            Initial state in normalized CR3BP coordinates
        """
        R_earth = 6378.0  # Earth radius (km)
        r0 = R_earth + altitude_km

        # Calculate departure velocity for given C3
        # v² = μ/r + C3
        v_departure = np.sqrt(self.constants.MU_EARTH / r0 + c3_energy)

        # Convert inclination to radians
        inc_rad = np.radians(inclination_deg)

        # Initial position (in Earth-centered frame)
        x_ec = r0 * np.cos(inc_rad)
        y_ec = 0.0
        z_ec = r0 * np.sin(inc_rad)

        # Initial velocity (inject towards Moon)
        vx_ec = 0.0
        vy_ec = v_departure * 0.95  # Most velocity in y-direction
        vz_ec = v_departure * 0.1   # Small out-of-plane component

        # Transform to CR3BP rotating frame
        # Earth is at (-μ, 0, 0) in rotating frame
        x_rot = (x_ec / self.constants.R_EARTH_MOON) - self.constants.mu
        y_rot = y_ec / self.constants.R_EARTH_MOON
        z_rot = z_ec / self.constants.R_EARTH_MOON

        vx_rot = vx_ec / self.constants.v_char + y_rot  # Add rotation component
        vy_rot = vy_ec / self.constants.v_char - x_rot  # Add rotation component
        vz_rot = vz_ec / self.constants.v_char

        return np.array([x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot])

    def simulate_mission(
        self,
        duration_days: float = 10.0,
        initial_altitude_km: float = 300.0,
        inclination_deg: float = 33.0
    ) -> Dict:
        """
        Simulate complete Artemis II-like mission

        Args:
            duration_days: Mission duration (days)
            initial_altitude_km: LEO parking orbit altitude
            inclination_deg: Orbital inclination

        Returns:
            Complete simulation results with diagnostics
        """
        # Generate initial conditions
        initial_state = self.create_free_return_initial_conditions(
            altitude_km=initial_altitude_km,
            inclination_deg=inclination_deg,
            c3_energy=-0.4  # Typical for lunar free-return
        )

        # Convert duration to normalized time
        t_normalized = (duration_days * 86400) / self.constants.t_char

        print(f"Starting Artemis II trajectory simulation...")
        print(f"Initial Jacobi constant: {self.dynamics.jacobi_constant(initial_state):.6f}")
        print(f"Duration: {duration_days} days ({t_normalized:.3f} normalized time units)")

        # Integrate trajectory
        result = self.integrator.integrate_trajectory(
            initial_state=initial_state,
            t_span=(0, t_normalized),
            method='DOP853',  # Dormand-Prince 8(5,3) - high accuracy
            rtol=1e-12,
            atol=1e-12,
            max_step=0.005
        )

        # Error analysis
        error_metrics = self.integrator.error_analysis(result)

        # Check if trajectory returns to Earth vicinity
        final_state = result['state'][:, -1]
        final_pos, final_vel = self.normalized_to_dimensional(final_state)

        # Distance from Earth at end
        r_earth_final = np.linalg.norm(final_pos +
                                       np.array([self.constants.mu * self.constants.R_EARTH_MOON, 0, 0]))

        is_free_return = r_earth_final < 10000.0  # Within 10,000 km of Earth

        print(f"\nSimulation complete!")
        print(f"Function evaluations: {result['n_evaluations']}")
        print(f"Maximum Jacobi error: {result['max_jacobi_error']:.2e}")
        print(f"Relative Jacobi error: {result['jacobi_relative_error']:.2e}")
        print(f"Final Earth distance: {r_earth_final:.1f} km")
        print(f"Free-return achieved: {is_free_return}")

        return {
            'integration_result': result,
            'error_metrics': error_metrics,
            'is_free_return': is_free_return,
            'final_earth_distance_km': r_earth_final,
            'initial_conditions': initial_state,
            'constants': self.constants
        }


class TrajectoryAnalyzer:
    """
    Analyze trajectory characteristics and validate physical consistency
    """

    @staticmethod
    def calculate_periapse_distances(
        state_history: np.ndarray,
        constants: PhysicalConstants
    ) -> Dict:
        """
        Find closest approach distances to Earth and Moon
        """
        mu = constants.mu

        # Distance to Earth (at -μ, 0, 0)
        earth_distances = np.sqrt(
            (state_history[0, :] + mu)**2 +
            state_history[1, :]**2 +
            state_history[2, :]**2
        ) * constants.R_EARTH_MOON

        # Distance to Moon (at 1-μ, 0, 0)
        moon_distances = np.sqrt(
            (state_history[0, :] - (1 - mu))**2 +
            state_history[1, :]**2 +
            state_history[2, :]**2
        ) * constants.R_EARTH_MOON

        return {
            'earth_perigee_km': np.min(earth_distances),
            'moon_periapse_km': np.min(moon_distances),
            'earth_perigee_idx': np.argmin(earth_distances),
            'moon_periapse_idx': np.argmin(moon_distances)
        }

    @staticmethod
    def validate_physical_consistency(mission_result: Dict) -> Dict[str, bool]:
        """
        Validate trajectory against physical constraints
        """
        result = mission_result['integration_result']
        error_metrics = mission_result['error_metrics']

        checks = {}

        # 1. Energy conservation (Jacobi constant drift)
        checks['energy_conserved'] = error_metrics['relative_jacobi_drift'] < 1e-6

        # 2. Numerical stability
        checks['numerically_stable'] = result['success']

        # 3. No collision with Earth (radius ~ 6378 km)
        R_earth = 6378.0
        earth_dist = TrajectoryAnalyzer.calculate_periapse_distances(
            result['state'],
            mission_result['constants']
        )['earth_perigee_km']
        checks['no_earth_collision'] = earth_dist > R_earth + 100  # 100 km buffer

        # 4. No collision with Moon (radius ~ 1737 km)
        R_moon = 1737.0
        moon_dist = TrajectoryAnalyzer.calculate_periapse_distances(
            result['state'],
            mission_result['constants']
        )['moon_periapse_km']
        checks['no_moon_collision'] = moon_dist > R_moon + 100  # 100 km buffer

        # 5. Free-return condition
        checks['free_return_achieved'] = mission_result['is_free_return']

        return checks


def export_trajectory_data(mission_result: Dict, filename: str = "artemis_trajectory.json"):
    """
    Export trajectory data for visualization
    """
    result = mission_result['integration_result']
    constants = mission_result['constants']

    # Convert to dimensional units
    time_days = result['time'] * constants.t_char / 86400

    # Get validation results if available
    validation = mission_result.get('validation', {})
    validation_serializable = {k: bool(v) for k, v in validation.items()} if validation else {}

    # Convert error_metrics to serializable format
    error_metrics_serializable = {}
    for k, v in mission_result['error_metrics'].items():
        if isinstance(v, (tuple, list)):
            error_metrics_serializable[k] = [float(x) for x in v]
        elif isinstance(v, (bool, np.bool_)):
            error_metrics_serializable[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            error_metrics_serializable[k] = int(v)
        else:
            error_metrics_serializable[k] = float(v)

    data = {
        'time_days': time_days.tolist(),
        'state_normalized': result['state'].tolist(),
        'jacobi_constant': float(result['jacobi_initial']),
        'jacobi_error': result['jacobi_error'].tolist(),
        'error_metrics': error_metrics_serializable,
        'is_free_return': bool(mission_result['is_free_return']),
        'validation': validation_serializable,
        'constants': {
            'mu_earth': float(constants.MU_EARTH),
            'mu_moon': float(constants.MU_MOON),
            'r_earth_moon': float(constants.R_EARTH_MOON),
            'mass_ratio': float(constants.mu)
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Trajectory data exported to {filename}")


if __name__ == "__main__":
    # Run Artemis II simulation
    artemis = ArtemisIITrajectory()

    print("=" * 60)
    print("ARTEMIS II FREE-RETURN TRAJECTORY SIMULATION")
    print("Circular Restricted Three-Body Problem")
    print("=" * 60)
    print()

    # Simulate 10-day mission
    mission_result = artemis.simulate_mission(
        duration_days=10.0,
        initial_altitude_km=300.0,
        inclination_deg=33.0
    )

    # Analyze trajectory
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS")
    print("=" * 60)

    periapse = TrajectoryAnalyzer.calculate_periapse_distances(
        mission_result['integration_result']['state'],
        mission_result['constants']
    )

    print(f"\nClosest approaches:")
    print(f"  Earth perigee: {periapse['earth_perigee_km']:.1f} km")
    print(f"  Moon periapse: {periapse['moon_periapse_km']:.1f} km")

    # Physical validation
    print("\n" + "=" * 60)
    print("PHYSICAL CONSISTENCY CHECKS")
    print("=" * 60)

    validation = TrajectoryAnalyzer.validate_physical_consistency(mission_result)
    mission_result['validation'] = validation  # Add to mission result
    for check, passed in validation.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {check}: {status}")

    # Export data
    print("\n" + "=" * 60)
    export_trajectory_data(mission_result, "artemis_ii_trajectory.json")
    print("=" * 60)
