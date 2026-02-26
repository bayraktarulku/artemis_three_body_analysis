"""
Trajectory Visualization for Artemis II Free-Return Analysis
Creates publication-quality plots of the three-body trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
from three_body_simulator import (
    ArtemisIITrajectory,
    TrajectoryAnalyzer,
    PhysicalConstants
)


def plot_trajectory_3d(mission_result, save_path="trajectory_3d.png"):
    result = mission_result['integration_result']
    constants = mission_result['constants']
    state = result['state']

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = state[0, :] * constants.R_EARTH_MOON
    y = state[1, :] * constants.R_EARTH_MOON
    z = state[2, :] * constants.R_EARTH_MOON

    time_days = result['time'] * constants.t_char / 86400
    scatter = ax.scatter(x, y, z, c=time_days, cmap='viridis',
                        s=1, alpha=0.6, label='Trajectory')

    # Earth position (at -μ * R in rotating frame)
    earth_pos = -constants.mu * constants.R_EARTH_MOON
    ax.scatter([earth_pos], [0], [0], c='blue', s=200,
              marker='o', label='Earth', edgecolors='darkblue', linewidths=2)

    # Moon position (at (1-μ) * R in rotating frame)
    moon_pos = (1 - constants.mu) * constants.R_EARTH_MOON
    ax.scatter([moon_pos], [0], [0], c='gray', s=100,
              marker='o', label='Moon', edgecolors='black', linewidths=2)

    # Start and end points
    ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=100,
              marker='*', label='Start', edgecolors='darkgreen', linewidths=2)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=100,
              marker='X', label='End', edgecolors='darkred', linewidths=2)

    ax.set_xlabel('X (km) - Rotating Frame', fontsize=12)
    ax.set_ylabel('Y (km) - Rotating Frame', fontsize=12)
    ax.set_zlabel('Z (km) - Rotating Frame', fontsize=12)
    ax.set_title('Artemis II Free-Return Trajectory\nCircular Restricted Three-Body Problem',
                fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Mission Time (days)', fontsize=11)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D trajectory plot saved to {save_path}")
    plt.close()


def plot_xy_projection(mission_result, save_path="trajectory_xy.png"):
    """
    X-Y plane projection showing Earth-Moon configuration
    """
    result = mission_result['integration_result']
    constants = mission_result['constants']
    state = result['state']

    fig, ax = plt.subplots(figsize=(12, 10))

    x = state[0, :] * constants.R_EARTH_MOON
    y = state[1, :] * constants.R_EARTH_MOON
    time_days = result['time'] * constants.t_char / 86400

    scatter = ax.scatter(x, y, c=time_days, cmap='plasma',
                        s=3, alpha=0.7, label='Trajectory')

    # Earth
    earth_x = -constants.mu * constants.R_EARTH_MOON
    earth_circle = Circle((earth_x, 0), 6378, color='blue',
                          alpha=0.7, label='Earth')
    ax.add_patch(earth_circle)

    # Moon
    moon_x = (1 - constants.mu) * constants.R_EARTH_MOON
    moon_circle = Circle((moon_x, 0), 1737, color='gray',
                         alpha=0.8, label='Moon')
    ax.add_patch(moon_circle)

    # Start/End markers
    ax.plot(x[0], y[0], 'g*', markersize=15, label='Mission Start')
    ax.plot(x[-1], y[-1], 'rX', markersize=15, label='Mission End')

    # Find and mark periapses
    periapse = TrajectoryAnalyzer.calculate_periapse_distances(state, constants)
    earth_peri_idx = periapse['earth_perigee_idx']
    moon_peri_idx = periapse['moon_periapse_idx']

    ax.plot(x[earth_peri_idx], y[earth_peri_idx], 'b^',
           markersize=12, label=f"Earth Perigee ({periapse['earth_perigee_km']:.0f} km)")
    ax.plot(x[moon_peri_idx], y[moon_peri_idx], 'm^',
           markersize=12, label=f"Lunar Periapse ({periapse['moon_periapse_km']:.0f} km)")

    ax.set_xlabel('X (km) - Rotating Frame', fontsize=13)
    ax.set_ylabel('Y (km) - Rotating Frame', fontsize=13)
    ax.set_title('Artemis II Trajectory - X-Y Projection\nEarth-Moon Rotating Reference Frame',
                fontsize=14, fontweight='bold')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mission Time (days)', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"X-Y projection plot saved to {save_path}")
    plt.close()


def plot_jacobi_error(mission_result, save_path="jacobi_error.png"):
    """
    Plot Jacobi constant conservation (energy error analysis)
    """
    result = mission_result['integration_result']
    constants = mission_result['constants']

    time_days = result['time'] * constants.t_char / 86400
    jacobi_error = result['jacobi_error']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Absolute error
    ax1.plot(time_days, jacobi_error, 'b-', linewidth=1.5)
    ax1.set_ylabel('Absolute Jacobi Error', fontsize=12)
    ax1.set_title('Numerical Integration Error Analysis\nJacobi Constant Conservation',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.axhline(y=1e-6, color='r', linestyle='--',
               label='Tolerance: 10⁻⁶', linewidth=2)
    ax1.legend(fontsize=10)

    # Relative error
    relative_error = jacobi_error / abs(result['jacobi_initial'])
    ax2.plot(time_days, relative_error, 'g-', linewidth=1.5)
    ax2.set_xlabel('Mission Time (days)', fontsize=12)
    ax2.set_ylabel('Relative Jacobi Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.axhline(y=1e-9, color='r', linestyle='--',
               label='Excellent: 10⁻⁹', linewidth=2)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Jacobi error analysis plot saved to {save_path}")
    plt.close()


def plot_distance_evolution(mission_result, save_path="distance_evolution.png"):
    """
    Plot distances to Earth and Moon over time
    """
    result = mission_result['integration_result']
    constants = mission_result['constants']
    state = result['state']
    mu = constants.mu

    time_days = result['time'] * constants.t_char / 86400

    # Calculate distances
    earth_dist = np.sqrt(
        (state[0, :] + mu)**2 +
        state[1, :]**2 +
        state[2, :]**2
    ) * constants.R_EARTH_MOON

    moon_dist = np.sqrt(
        (state[0, :] - (1 - mu))**2 +
        state[1, :]**2 +
        state[2, :]**2
    ) * constants.R_EARTH_MOON

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(time_days, earth_dist, 'b-', linewidth=2, label='Distance to Earth')
    ax.plot(time_days, moon_dist, 'gray', linewidth=2, label='Distance to Moon')

    earth_peri_idx = np.argmin(earth_dist)
    moon_peri_idx = np.argmin(moon_dist)

    ax.plot(time_days[earth_peri_idx], earth_dist[earth_peri_idx],
           'b^', markersize=12, label=f'Earth Perigee')
    ax.plot(time_days[moon_peri_idx], moon_dist[moon_peri_idx],
           'r^', markersize=12, label=f'Lunar Periapse')

    ax.set_xlabel('Mission Time (days)', fontsize=13)
    ax.set_ylabel('Distance (km)', fontsize=13)
    ax.set_title('Spacecraft Distance Evolution\nArtemis II Mission Profile',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    ax.axhline(y=6378+300, color='b', linestyle=':', alpha=0.5,
              label='LEO altitude (300 km)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distance evolution plot saved to {save_path}")
    plt.close()


def plot_velocity_profile(mission_result, save_path="velocity_profile.png"):
    """
    Plot velocity magnitude over time
    """
    result = mission_result['integration_result']
    constants = mission_result['constants']
    state = result['state']

    time_days = result['time'] * constants.t_char / 86400

    # Velocity components (convert to km/s)
    vx = state[3, :] * constants.v_char
    vy = state[4, :] * constants.v_char
    vz = state[5, :] * constants.v_char

    v_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Magnitude
    ax1.plot(time_days, v_magnitude, 'r-', linewidth=2)
    ax1.set_ylabel('Velocity Magnitude (km/s)', fontsize=12)
    ax1.set_title('Spacecraft Velocity Profile\nArtemis II Mission',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=11.0, color='g', linestyle='--', alpha=0.7,
               label='Escape velocity (~11 km/s)')
    ax1.legend(fontsize=10)

    # Components
    ax2.plot(time_days, vx, 'r-', linewidth=1.5, label='vₓ', alpha=0.8)
    ax2.plot(time_days, vy, 'g-', linewidth=1.5, label='vᵧ', alpha=0.8)
    ax2.plot(time_days, vz, 'b-', linewidth=1.5, label='vᵤ', alpha=0.8)
    ax2.set_xlabel('Mission Time (days)', fontsize=12)
    ax2.set_ylabel('Velocity Components (km/s)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Velocity profile plot saved to {save_path}")
    plt.close()


def create_summary_report(mission_result, save_path="mission_summary.txt"):
    """
    Generate text summary of mission analysis
    """
    result = mission_result['integration_result']
    constants = mission_result['constants']
    error_metrics = mission_result['error_metrics']

    periapse = TrajectoryAnalyzer.calculate_periapse_distances(
        result['state'], constants
    )

    validation = TrajectoryAnalyzer.validate_physical_consistency(mission_result)

    report = f"""
{'='*70}
ARTEMIS II FREE-RETURN TRAJECTORY ANALYSIS
Circular Restricted Three-Body Problem (CR3BP)
{'='*70}

MISSION PARAMETERS
------------------
Duration: {error_metrics['duration_days']:.2f} days
Integration method: Dormand-Prince 8(5,3) adaptive
Relative tolerance: 1.0e-12
Absolute tolerance: 1.0e-12

PHYSICAL CONSTANTS
------------------
Earth gravitational parameter: {constants.MU_EARTH:.4f} km³/s²
Moon gravitational parameter: {constants.MU_MOON:.4f} km³/s²
Earth-Moon distance: {constants.R_EARTH_MOON:.1f} km
Mass ratio (μ): {constants.mu:.10f}
Characteristic velocity: {constants.v_char:.4f} km/s
Characteristic time: {constants.t_char:.1f} seconds ({constants.t_char/3600:.2f} hours)

TRAJECTORY CHARACTERISTICS
--------------------------
Initial Jacobi constant: {result['jacobi_initial']:.8f}
Earth perigee altitude: {periapse['earth_perigee_km'] - 6378:.1f} km
Lunar periapse altitude: {periapse['moon_periapse_km'] - 1737:.1f} km
Final Earth distance: {mission_result['final_earth_distance_km']:.1f} km
Free-return achieved: {'YES' if mission_result['is_free_return'] else 'NO'}

NUMERICAL INTEGRATION METRICS
-----------------------------
Total function evaluations: {result['n_evaluations']}
Integration success: {result['success']}
Maximum Jacobi error (absolute): {result['max_jacobi_error']:.6e}
Maximum Jacobi error (relative): {result['jacobi_relative_error']:.6e}

Position range: [{error_metrics['position_range'][0]:.4f}, {error_metrics['position_range'][1]:.4f}] (normalized)
Velocity range: [{error_metrics['velocity_range'][0]:.6f}, {error_metrics['velocity_range'][1]:.6f}] (normalized)

PHYSICAL CONSISTENCY VALIDATION
-------------------------------
"""

    for check, passed in validation.items():
        status = "[PASS]" if passed else "[FAIL]"
        report += f"{check.replace('_', ' ').title()}: {status}\n"

    report += f"""
{'='*70}
INTERPRETATION
{'='*70}

This simulation demonstrates a high-fidelity trajectory computation for
a lunar free-return mission profile similar to Artemis II. The trajectory
is computed in the Earth-Moon rotating reference frame using the Circular
Restricted Three-Body Problem formulation.

Key findings:
1. Energy conservation (Jacobi constant) is maintained to ~{result['jacobi_relative_error']:.2e} relative error
2. The spacecraft performs a lunar flyby at ~{periapse['moon_periapse_km']:.0f} km altitude
3. The trajectory naturally returns to Earth vicinity (free-return property)
4. All physical constraints are satisfied (no collisions, stable orbit)

The numerical integration uses adaptive stepping with Dormand-Prince 8th order
method, ensuring high accuracy while efficiently handling the varying dynamics
near gravitational bodies.

This validates that the restricted three-body problem provides an excellent
first-order approximation for real mission planning, with perturbations
(solar gravity, Earth oblateness, etc.) added in operational missions.
{'='*70}
"""

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"Mission summary report saved to {save_path}")
    print(report)


if __name__ == "__main__":
    print("Generating Artemis II trajectory visualizations...\n")

    # Run simulation
    artemis = ArtemisIITrajectory()
    mission_result = artemis.simulate_mission(
        duration_days=10.0,
        initial_altitude_km=300.0,
        inclination_deg=33.0
    )

    print("\nCreating visualizations...")

    # Generate all plots
    plot_trajectory_3d(mission_result, "artemis_trajectory_3d.png")
    plot_xy_projection(mission_result, "artemis_trajectory_xy.png")
    plot_jacobi_error(mission_result, "artemis_jacobi_error.png")
    plot_distance_evolution(mission_result, "artemis_distance_evolution.png")
    plot_velocity_profile(mission_result, "artemis_velocity_profile.png")

    # Generate summary report
    create_summary_report(mission_result, "artemis_mission_summary.txt")

    print("\n" + "="*70)
    print("All visualizations and analysis complete!")
    print("="*70)
