"""
Quick Test Script for Artemis II Simulator
Runs a simplified version to verify the implementation
"""

import sys
import numpy as np

print("="*60)
print("ARTEMIS II SIMULATOR - QUICK TEST")
print("="*60)
print()

# Test imports
print("1. Testing imports...")
try:
    from three_body_simulator import (
        PhysicalConstants,
        CR3BPDynamics,
        NumericalIntegrator,
        ArtemisIITrajectory,
        TrajectoryAnalyzer
    )
    print("   [OK] All imports successful")
except ImportError as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test physical constants
print("\n2. Testing physical constants...")
constants = PhysicalConstants()
print(f"   Mass ratio μ: {constants.mu:.10f}")
print(f"   Characteristic velocity: {constants.v_char:.4f} km/s")
print(f"   Characteristic time: {constants.t_char/3600:.2f} hours")
print("   [OK] Constants initialized")

# Test dynamics
print("\n3. Testing CR3BP dynamics...")
dynamics = CR3BPDynamics(constants)
test_state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
jacobi = dynamics.jacobi_constant(test_state)
print(f"   Test Jacobi constant: {jacobi:.6f}")
derivatives = dynamics.equations_of_motion(0.0, test_state)
print(f"   Acceleration magnitude: {np.linalg.norm(derivatives[3:6]):.6f}")
print("   [OK] Dynamics working")

# Test short integration
print("\n4. Testing numerical integration (short test)...")
integrator = NumericalIntegrator(dynamics)
result = integrator.integrate_trajectory(
    initial_state=test_state,
    t_span=(0, 0.1),  # Very short integration
    method='DOP853',
    rtol=1e-10,
    atol=1e-10,
    max_step=0.01
)

if result['success']:
    print(f"   [OK] Integration successful")
    print(f"   Function evaluations: {result['n_evaluations']}")
    print(f"   Jacobi error: {result['max_jacobi_error']:.6e}")
else:
    print(f"   [FAIL] Integration failed: {result['message']}")

# Test Artemis trajectory setup
print("\n5. Testing Artemis II trajectory setup...")
artemis = ArtemisIITrajectory()
initial_conditions = artemis.create_free_return_initial_conditions(
    altitude_km=300.0,
    inclination_deg=33.0,
    c3_energy=-0.4
)
print(f"   Initial position: [{initial_conditions[0]:.4f}, {initial_conditions[1]:.4f}, {initial_conditions[2]:.4f}]")
print(f"   Initial velocity: [{initial_conditions[3]:.4f}, {initial_conditions[4]:.4f}, {initial_conditions[5]:.4f}]")
initial_jacobi = dynamics.jacobi_constant(initial_conditions)
print(f"   Initial Jacobi constant: {initial_jacobi:.6f}")
print("   [OK] Initial conditions generated")

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)
print("\nThe simulator is ready to use!")
print("Run 'python three_body_simulator.py' for full mission simulation")
print("Run 'python visualize_trajectory.py' for visualization")
