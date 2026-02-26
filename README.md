# Artemis II Three-Body Problem Analysis

Artemis II free-return trajectory simulation using CR3BP (Circular Restricted Three-Body Problem).

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
cd artemis_three_body_analysis
uv run python test_simulator.py

# Run simulation
uv run python three_body_simulator.py

# Generate visualizations
uv run python visualize_trajectory.py
```

## Test Results

### Validation Tests
```
[OK] All imports successful
[OK] Physical constants initialized (μ = 0.0121505839)
[OK] CR3BP dynamics verified (test Jacobi: 3.192041)
[OK] Numerical integration stable (Jacobi error: 8.88e-16)
[OK] Initial conditions generated correctly
```

### Simulation Results
```
Duration:                10.0 days
Function Evaluations:    123,155
Jacobi Error (relative): 1.78e-10
Jacobi Error (absolute): 1.11e-08
Earth Perigee:           5,469 km
Moon Periapsis:          378,005 km
Final Earth Distance:    6,709 km

Validation:
  Energy Conservation:   [PASS]
  Numerical Stability:   [PASS]
  No Moon Collision:     [PASS]
  Free Return Achieved:  [PASS]
```

## Outputs

- `artemis_trajectory_3d.png` - 3D trajectory visualization
- `artemis_trajectory_xy.png` - X-Y projection (rotating frame)
- `artemis_jacobi_error.png` - Energy conservation analysis
- `artemis_velocity_profile.png` - Velocity profile
- `artemis_mission_summary.txt` - Detailed report

## Technical Details

- **Numerical Method:** Dormand-Prince 8(5,3) adaptive stepping
- **Tolerance:** 10⁻¹² (relative and absolute)
- **Energy Conservation:** 1.78×10⁻¹⁰ relative error
- **Model:** Circular Restricted Three-Body Problem (CR3BP)
- **Integration Time:** 10 days (~123k function evaluations)

## Articles

Full technical explanations are available on Medium:

**Turkish Article:**  
- Medium: _(coming soon)_
- Source: [`MEDIUM_ARTICLE_TR.md`](artemis_three_body_analysis/MEDIUM_ARTICLE_TR.md)

**English Article:**  
- Medium: _(coming soon)_
- Source: [`MEDIUM_ARTICLE_EN.md`](artemis_three_body_analysis/MEDIUM_ARTICLE_EN.md)

---

## License

MIT License - This is a hobby project for educational purposes.

**Note:** This is a simplified CR3BP model. Real mission design includes solar gravity, Earth oblateness, lunar mascons, and other perturbations.
