# Artemis II and the Three-Body Problem: A Developer's Trajectory Notes

Artemis II, scheduled for launch in April 2026, will be the first crewed mission beyond low Earth orbit since Apollo 17. Four astronauts will loop around the Moon and return. No landing. No flag planting. But from a dynamics perspective, it's quite sophisticated: a system launched at the right energy level, right phase, right geometry—falling back to Earth after looping behind the Moon, under gravity alone.

This isn't a formal guide. Think of it as notes taken while learning. That urge you feel while coding: "Can I simulate this myself?" Questions lead to answers; answers lead to more questions. There may be gaps or errors—corrections are welcome.

My focus: the mathematics behind free-return trajectories. Specifically, the Circular Restricted Three-Body Problem (CR3BP) and the numerical methods used to solve it.

---

## The Three-Body Problem: Deterministic But Unsolvable

Newton's law of gravitation (1687) made the two-body problem analytically solvable. In a two-mass system like Earth and Sun, motion can be expressed in closed form. Ellipses, hyperbolas, parabolas… Clean.

Add a third body and things get messy. The system is still deterministic—if you know the initial conditions exactly, the future is determined. But there's no general analytical solution. Poincaré showed why in the late 19th century: the system is chaotic. Small differences can lead to dramatic divergence over time.

Space missions are built on these "unsolvable but computable" dynamics.

---

## The Circular Restricted Three-Body Problem (CR3BP)

The model used to understand Artemis II's Earth–Moon dynamics as a first approximation is the Circular Restricted Three-Body Problem (CR3BP).

The simplifications:

- Earth and Moon orbit each other in a circular path.
- The spacecraft's mass is negligible (doesn't affect Earth or Moon).
- The coordinate system rotates with the Earth–Moon line.
- Solar gravity and other perturbations are initially ignored.

This model is physically incomplete but structurally powerful. It lets us see the system's topology: which regions are accessible at which energy levels, which gates are open, which are closed.

---

## Rotating Reference Frame and Geometry

We choose a coordinate system that rotates with angular velocity along the Earth–Moon line. In this frame:

- Earth is at a fixed point,
- Moon is at a fixed point,
- Spacecraft moves.

This choice matters. In an inertial frame, Earth and Moon are constantly moving; here, the problem becomes geometrically simpler.

In normalized units, we set the Earth–Moon distance to 1. The mass ratio:

```
μ = M_Moon / (M_Earth + M_Moon) ≈ 0.01215
```

This small number represents the system's asymmetry. Earth dominates, but the Moon cannot be ignored.

The spacecraft's position vector **r** = (x, y, z), velocity vector **v** = (vₓ, vᵧ, vᵤ). The equations of motion reduce to a six-dimensional first-order differential equation system.

The key concept here is the "effective potential":

```
U(x,y,z) = (1-μ)/r₁ + μ/r₂ + ½(x² + y²)
```

The last term represents the centrifugal effect. This is the cost of the rotating reference frame: Coriolis and centrifugal terms are added.

At this point, the system is "integration-ready." No analytical solution, but numerical solution is possible.

![Artemis II Trajectory - Rotating Frame](artemis_trajectory_xy.png)
*Figure 2: Artemis II trajectory in rotating coordinate system (xy plane). Earth and Moon appear stationary. Trajectory starts from Earth, loops behind the Moon, and returns.*

---

## The Jacobi Constant: An Energy-Like Invariant

One of CR3BP's most elegant features is the Jacobi constant:

```
C = 2U - v²
```

This quantity remains constant along the trajectory. It's similar to total energy in classical mechanics but specific to the rotating reference frame.

The Jacobi constant tells us: at what "energy level" is the spacecraft moving? This energy level determines accessible regions. Surfaces called zero-velocity surfaces define boundaries the spacecraft cannot cross.

For example:

- High C → orbit confined to Earth vicinity
- C ≈ 3 range → passages near Lagrange points possible
- Lower C → Moon access possible

Free-return trajectories occur within a specific Jacobi range. So the issue isn't "going to the Moon"; it's settling into the right energy topology.

![Jacobi Constant Error Analysis](artemis_jacobi_error.png)
*Figure 1: Jacobi constant error analysis over 10-day simulation. Error spike visible during lunar flyby (~day 5), but overall stability is maintained.*

---

## Numerical Integration: The Real Work Begins

We have the differential equations. Now the question: how do we integrate these for 10 days while controlling error accumulation?

The Runge–Kutta family comes into play. Specifically, high-order methods with adaptive step control.

For precision missions like Artemis II, methods like Dormand–Prince 8(5,3) (DOP853) are preferred. This method:

- Provides 8th-order accuracy
- Estimates error with embedded 5th-order solution
- Adjusts step size dynamically

The basic idea is simple:

1. Calculate derivative from current state.
2. Estimate next step.
3. Compare two different order solutions.
4. If error exceeds tolerance, reject step and reduce size.
5. If acceptable, accept and continue.

Local error ~ O(h⁹), global error ~ O(h⁸).

But the order on paper isn't enough alone. The real test: is the Jacobi constant conserved?

---

## Error Dynamics and Lunar Flyby

In simulations, you'll notice something interesting: Jacobi constant error increases during close lunar approach. This is because the gravity gradient changes rapidly. The system isn't stiff, but it's sensitive.

Adaptive step control plays a critical role here. As you approach the Moon, step size decreases. As you move away, it increases again.

In a 10-day integration, keeping Jacobi error at the 10⁻⁹ level shows that both the model and numerical method are working properly.

This is where developer instincts kick in:
- "Is there energy drift?"
- "Am I approaching a singularity?"
- "What's the step rejection rate?"

Code matters as much as physics.

---

## What is a Free-Return Trajectory?

Free-return is a ballistic trajectory that loops around the Moon and returns to Earth without using engines. It was designed as a safety mechanism during the Apollo era.

Advantages:

- Return even with engine failure.
- Fuel savings.
- Exploits natural dynamics.

Disadvantages:

- Geometry is constrained.
- Not suitable for lunar landing.
- Launch timing is critical.

In this trajectory, the Moon "bends" the spacecraft like a gravitational slingshot, redirecting it back. But this differs from a classic gravity assist; it's about direction change, not energy gain.

From a CR3BP perspective, the vehicle enters a phase space region accessible via L1/L2 gates at a specific Jacobi level, loops behind the Moon, and falls back into Earth's potential well.

This "falling back" isn't romantic—it's mathematically accurate.

---

## Simulation Outputs and Physical Checks

In a typical 10-day free-return simulation:

- Earth perigee: ~400 km
- Lunar periapsis: several hundred kilometers
- Maximum distance: ~380,000 km

![Artemis II 3D Trajectory](artemis_trajectory_3d.png)
*Figure 3: Three-dimensional view of Artemis II trajectory. Earth near center, Moon fixed on the right in rotating reference frame. Trajectory proceeds in a slightly inclined plane (~1°).*

Checklist:

- Jacobi constant drift < 10⁻⁶ → acceptable
- No descent below Earth or Moon radius → no collision
- Integration completed successfully → no divergence

Software-side metrics:

- Function call count
- Step rejection rate
- Minimum step size

These matter as much as physical results. A buggy integrator can produce a "correct-looking but wrong" trajectory.

---

## Model Limitations

CR3BP is powerful but incomplete.

What's ignored:

- Solar gravity
- Earth's J₂ term (equatorial bulge)
- Lunar mascons
- Solar radiation pressure

In real mission design, the process goes:

1. CR3BP for initial draft.
2. Real ephemeris data.
3. Add perturbations.
4. High-fidelity integrators.
5. Monte Carlo uncertainty analysis.

A mission like Artemis II is optimized through thousands of simulations.

CR3BP isn't the end; it's the beginning.

---

## Broader Perspective: Lagrange Points and the Future

This model also opens the door to understanding Lagrange points. Halo orbits around L1 and L2 are the foundation for projects like Lunar Gateway.

For Artemis III and beyond, lunar surface landing, NRHO (Near-Rectilinear Halo Orbit), and low-energy transfers will come into play. These are all extensions of three-body dynamics.

So the differential equations we're learning here shape not just a Moon trip—but the future cislunar economy.

---

## Conclusion: Determinism, Numerical Precision, and Reality

The three-body problem sits in an interesting place. The system is completely deterministic, but has no general solution. The future is determined, but cannot be written as a formula.

This resembles the nature of engineering.

CR3BP teaches us:

- The right abstraction makes complexity manageable.
- Numerical methods are extensions of theory.
- Energy conservation is a simulation's conscience.

Artemis II's free-return trajectory isn't a romantic "Moon tour"; it's an initial condition carefully chosen on energy surfaces.

As a developer, the most striking part for me:
A six-dimensional state vector, a few differential equations, and a well-written integrator… And at the end, a spacecraft that loops behind the Moon and comes home.

Deterministic but sensitive. Simple assumptions but complex results.

Perhaps there's an unexpected parallel between the three-body problem and human life:
Knowing initial conditions exactly is impossible, but with the right model, we can understand the direction.

---

## References

### Academic References

1. **Szebehely, V.** (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*. Academic Press.  
   → The classic source for the three-body problem.

2. **Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D.** (2000). *Dynamical Systems, the Three-Body Problem and Space Mission Design*.  
   → Application of CR3BP in space missions.

3. **Hairer, E., Nørsett, S. P., & Wanner, G.** (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer.  
   → Runge-Kutta methods and numerical integration.

4. **Dormand, J. R., & Prince, P. J.** (1980). "A family of embedded Runge-Kutta formulae." *Journal of Computational and Applied Mathematics*, 6(1), 19-26.  
   → Original publication of the DOP853 method.

5. **NASA** (2024). *Artemis II Mission Overview*.  
   → https://www.nasa.gov/artemis-ii

### Code and Simulation

All simulations discussed in this article are available as open-source Python code on GitHub:

**https://github.com/bayraktarulku/artemis_three_body_analysis**

In the repo:
- Full CR3BP formulation
- Dormand-Prince 8(5,3) integrator
- Jacobi constant analysis
- Test and validation scripts
- Visualization tools
- Artemis II reference trajectory

---

**February 2026**

Compiled from personal simulation notes made with NumPy, SciPy, and Matplotlib.

