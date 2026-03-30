# CFMZNeedleDSMC — Documentation

Particle-based DSMC solver for the **CFMZ kinetic equation** governing
needle-like (oriented rigid rod) particles.  This document covers the
mathematical model, numerical methods, physical observables, configuration
options, and the full test suite.

---

## Table of contents

1. [The CFMZ kinetic equation](#1-the-cfmz-kinetic-equation)
2. [Particle degrees of freedom](#2-particle-degrees-of-freedom)
3. [Time integration — Strang splitting](#3-time-integration--strang-splitting)
4. [Transport substep](#4-transport-substep)
5. [Mean-field Vlasov force](#5-mean-field-vlasov-force)
6. [Collision operators](#6-collision-operators)
7. [The Onsager potential and the isotropic–nematic transition](#7-the-onsager-potential-and-the-isotropicnematic-transition)
8. [Isothermal simulations — Andersen thermostat](#8-isothermal-simulations--andersen-thermostat)
9. [Diagnostics and observables](#9-diagnostics-and-observables)
10. [Configuration reference](#10-configuration-reference)
11. [Test index](#11-test-index)
12. [Output format](#12-output-format)

---

## 1. The CFMZ kinetic equation

The CFMZ equation (Carrillo–Farrell–Medaglia–Zerbinati) describes the
evolution of the one-particle distribution function f(t, **v**, θ, ω) for an
ensemble of 2D rigid rods in a spatially homogeneous setting:

```
∂f/∂t  +  ω ∂f/∂θ  +  F[f](θ) ∂f/∂ω  =  Q[f, f]
```

where:

- `ω ∂f/∂θ` is the free-rotation (drift) term — orientation evolves at the
  current angular velocity.
- `F[f](θ) ∂f/∂ω` is the mean-field (Vlasov) torque — a self-consistent force
  derived from a pair interaction potential W(θ₁, θ₂).
- `Q[f, f]` is the rigid-rod binary collision operator, which redistributes
  kinetic energy between translational and rotational degrees of freedom while
  conserving linear momentum and angular momentum.

The spatial homogeneity means translational velocities **v** do not advect
particles through physical space; they only enter through the collision
operator (contact velocity at impact).

---

## 2. Particle degrees of freedom

Each particle carries four fields:

| Field | Symbol | Range | Meaning |
|-------|--------|-------|---------|
| translational velocity | **v** ∈ ℝ² | unbounded | centre-of-mass velocity |
| orientation | θ ∈ (0, 2π) | periodic | angle of the rod axis |
| angular velocity | ω ∈ ℝ | unbounded | rotation rate about the centre |
| weight | w | 1.0 (fixed) | computational weight (unused in current solver) |

The rod has length L, mass m, and moment of inertia I.  All particles on a
given MPI rank are stored as flat arrays inside a PETSc `DMSwarm`.

---

## 3. Time integration — Strang splitting

Each time step uses **second-order Strang (operator) splitting** between the
Vlasov (Hamiltonian) part and the collision part:

```
[DKD](dt/2) → collision(s) → [DKD](dt/2)
```

where DKD is itself a Störmer–Verlet sub-splitting of the free-rotation drift D
and the Vlasov kick K:

```
D(dt/4) · K(dt/2) · D(dt/4)  →  collision  →  D(dt/4) · K(dt/2) · D(dt/4)
```

This composition is second-order accurate and time-reversible for the
Vlasov part, and consistent with the Nanbu collision operator for the
stochastic part.  When a thermostat is active it is applied once, after
the collision sub-steps, inside the full step.

---

## 4. Transport substep

The **drift** sub-step advances the orientation by free rotation:

```
θ ← θ + ω · dt   (then wrapped back onto (0, 2π))
```

The **kick** sub-step applies the mean-field torque:

```
ω ← ω + F[f](θ) · dt
```

When transport is disabled (`opts["transport"] = False`) both sub-steps
are skipped and the simulation is a pure collision relaxation.

---

## 5. Mean-field Vlasov force

A user-supplied callable `vlasov_force(theta)` may be passed to the
constructor.  It receives the local orientation array (shape `(nlocal, 1)`)
and must return an array of the same shape containing the torque on each
particle.

The force is derived from a pair interaction potential W(θ₁, θ₂) as

```
F[f](θ) = −dV_eff/dθ,    V_eff(θ) = ∫ W(θ, θ') f(θ') dθ'
```

A **cloud-in-cell (CIC)** density estimator is used: each particle linearly
interpolates its weight onto the two nearest histogram bins, giving a
smooth density ρ on a uniform grid of `bins` cells covering [0, 2π).
The convolution V_eff = W ∗ ρ is then computed as a matrix–vector product
on that grid (O(bins²) per step, independent of the particle count N).

The same density estimator is used for the optional `interaction_energy`
callable,

```
E[ρ] = ∫∫ W(θ₁, θ₂) ρ(θ₁) ρ(θ₂) dθ₁ dθ₂,
```

which is recorded alongside the kinetic energy in the simulation history.

---

## 6. Collision operators

### 6.1 Rigid-rod impulse mechanics

For every accepted collision pair (i, j) the solver:

1. Draws a random impact angle ψ → contact normal **n** = (cos ψ, sin ψ).
2. Draws a random contact arm ℓ ∈ [0, L] along rod i; contact arm on rod j
   is set to L (tip contact).
3. Computes the relative contact velocity:

   ```
   V = (v_i − v_j) + ω_i r_i⊥ − ω_j r_j⊥
   ```

   where **r**⊥ is the arm vector rotated 90°.

4. Computes the impulse:

   ```
   J = −(V · n) / (2/m + (c_i² + c_j²)/I)
   ```

   with lever arms c_k = r_k × n.

5. Updates post-collisional velocities using restitution coefficients
   e_v (translational) and e_om (rotational):

   ```
   v_i' = v_i + (1+e_v) J/m · n
   v_j' = v_j − (1+e_v) J/m · n
   ω_i' = ω_i − (1+e_om) J c_i / I
   ω_j' = ω_j + (1+e_om) J c_j / I
   ```

Nearly-parallel pairs (|θ_i − θ_j| < `cutoff`) are near-singular for the
full rigid-rod denominator; they fall back to a spherical head-on impulse
using e_v only.

### 6.2 Maxwell (uniform) kernel

All pairs are equally likely to collide.  Exactly

```
M_col = floor(ν · N_local · dt / 2)
```

pairs are drawn uniformly without replacement each step.  This is the
classical **Nanbu (1980)** method.  The constraint ν · dt ≤ 1 must hold.

### 6.3 Hard-needle NTC kernel

The collision kernel for 2D calamitic needles (arXiv:2508.10744, Example B) is

```
W(Ξ₁, Ξ₂) = |g · n| · S(θ₁, θ₂),    S = L |sin(θ₁ − θ₂)|
```

where **g** is the effective contact velocity.  Near-parallel rods
(|sin(Δθ)| ≈ 0) have vanishing cross-section and are almost never selected,
unlike the uniform Maxwell kernel.

**Bird's No-Time-Counter (NTC)** acceptance–rejection is applied:

1. Draw `M_cand = floor(ν_max · N_local · dt / 2)` candidate pairs.
2. For each candidate compute w = |g·n| · L|sin(Δθ)|.
3. Accept with probability w / ν_max.
4. Update the running maximum: ν_max ← max(ν_max, max(w)).

The running maximum `_nu_max` is initialised to `opts["nu"]` and grows
monotonically.  A good initial estimate is ν ≈ L · v_max.

---

## 7. The Onsager potential and the isotropic–nematic transition

### 7.1 The Onsager pair potential

The **Onsager excluded-volume potential** for 2D needles is

```
W(θ₁, θ₂) = |sin(θ₁ − θ₂)|
```

Parallel or anti-parallel rods (|sin| = 0) have zero excluded volume and
hence minimum interaction energy, so this potential **favours alignment**.
The resulting Vlasov torque is

```
F(θ) = L² ∫ sign(sin(θ − θ')) cos(θ − θ') ρ(θ') dθ'
```

implemented via a discrete gradient of the CIC potential grid:

```
F(θ ∈ bin k) = L² (W_grid[k] − W_grid[k+1]) / Δθ
```

### 7.2 The isotropic–nematic (I–N) transition

The mean-field equilibrium distribution satisfies the Onsager
self-consistency equation

```
ρ(θ) ∝ exp(β L² ∫ |sin(θ − θ')| ρ(θ') dθ')
```

with β = 1/T.  The dimensionless **Onsager coupling** is

```
α = β L² = L² / T
```

Linear stability analysis of the isotropic solution ρ₀ = 1/(2π) with
respect to the nematic mode cos(2θ) gives the **spinodal condition**

```
α_c = 3π/2 ≈ 4.71
```

i.e. the critical temperature

```
T_c = L² · 2 / (3π)    [for L = √12 this gives T_c = 8/π ≈ 2.55]
```

The transition is weakly first-order; the actual coexistence temperature
lies close to T_c.

### 7.3 Microcanonical (NVE) vs. canonical (NVT) runs

In a **microcanonical** run (no thermostat) total energy E_kin + E[ρ] is
conserved.  As nematic order forms, E[ρ] decreases and E_kin increases,
so the temperature rises alongside the alignment.  This is what tests 8–12
and 21–24 observe.

In a **canonical** run (Andersen thermostat, see §8) the bath fixes T at
T_bath.  The system then explores the phase diagram at fixed temperature:
if α = L²/T_bath > α_c nematic order emerges; otherwise it remains isotropic.

---

## 8. Isothermal simulations — Andersen thermostat

### 8.1 Method

The **Andersen (1980) thermostat** couples particles to a heat bath at
temperature T_bath.  With frequency `nu_bath`, each particle independently
undergoes a "bath collision" in which its velocities are resampled from the
Maxwellian at T_bath:

```
v  ~ N(0, √(T_bath / m))   (each Cartesian component)
ω  ~ N(0, √(T_bath / I))
```

Each particle is resampled with probability `nu_bath · dt` per time step.
Orientations θ are never modified by the thermostat.

This is equivalent to adding a BGK relaxation term

```
Q_bath[f] = ν_bath (M_{T_bath} − f)
```

to the kinetic equation, where M_{T_bath} is the Maxwellian at T_bath.
It generates the **exact NVT (canonical) ensemble** at T_bath.

### 8.2 Effect on the phase diagram

With the thermostat active the long-time orientational distribution is the
canonical equilibrium

```
ρ(θ) ∝ exp(−β_bath V_eff(θ)),    β_bath = 1 / T_bath
```

which is exactly the Onsager self-consistency solution.  By scanning
T_bath one recovers the classical I–N phase diagram:

| T_bath | α = L²/T_bath | Phase |
|--------|--------------|-------|
| > T_c  | < α_c ≈ 4.71 | Isotropic (circular_var ≈ 1) |
| = T_c  | ≈ α_c | Critical (large fluctuations) |
| < T_c  | > α_c | Nematic (circular_var → 0) |

### 8.3 Enabling the thermostat

Pass `T_bath` and optionally `nu_bath` in `opts`:

```python
opts = {
    ...
    "T_bath":  0.5,   # target temperature (None = disabled, i.e. NVE)
    "nu_bath": 4.0,   # bath collision frequency (default 1.0)
}
```

Setting `nu_bath` comparable to `nu` (the physical collision frequency)
gives coupling strong enough to maintain T_bath on a timescale of a few
collision times.

---

## 9. Diagnostics and observables

### 9.1 Temperature

The kinetic temperature is defined from the total kinetic energy per
particle across all degrees of freedom:

```
T = (2 / (dim + 1)) · E_kin / N,    E_kin = Σ (½m|v|² + ½Iω²)
```

with dim = 2 (two translational modes + one rotational → three DOF,
factor 2/3).  An Andersen thermostat at T_bath drives T → T_bath.

### 9.2 Circular variance and nematic order parameter

The nematic order parameter is the mean resultant length of the doubled
angle:

```
R₂ = |N⁻¹ Σ exp(2i θ_k)|
```

The **circular variance on the real projective plane**

```
σ² = 1 − R₂
```

is the primary order parameter used by the CFMZ solver
(`variance = "real_projective_plane"`):

- σ² ≈ 1: **isotropic** phase (uniform orientation distribution)
- σ² ≈ 0: **nematic** phase (rods aligned along a common director)

The alternative (`variance = "circle"`) uses R₁ = |⟨exp(iθ)⟩| and
detects polar (head–tail asymmetric) ordering.

### 9.3 Interaction and total energy

When an `interaction_energy` callable is provided the history also
records:

```
E_int   = ∫∫ W(θ₁, θ₂) ρ(θ₁) ρ(θ₂) dθ₁ dθ₂
E_total = E_kin/N + ½ L² E_int
```

In a microcanonical run E_total is conserved.  Under the thermostat
E_kin/N → (3/2) T_bath while E_int evolves to its canonical average.

---

## 10. Configuration reference

### `opts` dictionary

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nlocal` | int | — | Particles per MPI rank |
| `nu` | float | 1.0 | Collision frequency (Maxwell) or initial NTC estimate (hard-needle) |
| `dt` | float | 0.01 | Time step; must satisfy ν·dt ≤ 1 for Maxwell kernel |
| `bins` | int | 31 | Histogram bins for the CIC density grid |
| `test` | str | `"uniform_angle"` | Initial condition: `"uniform_angle"` or `"perturbed_uniform_angle"` |
| `collision_type` | str | `"nanbu"` | Only `"nanbu"` is implemented |
| `extra_collision` | int | 1 | Collision sub-steps per time step |
| `variance` | str | `"circle"` | Order parameter geometry: `"circle"` or `"real_projective_plane"` |
| `seed` | int | 1234 | RNG seed (offset by MPI rank) |
| `prefix` | str | `""` | Path prefix for output directories |
| `transport` | bool | `True` | If `False`, skip drift and kick (pure collision) |
| `T_bath` | float or None | `None` | Andersen thermostat target temperature; `None` = NVE run |
| `nu_bath` | float | 1.0 | Bath collision frequency for the Andersen thermostat |
| `init_at_T_bath` | bool | `True` | When `True` and `T_bath` is set, initialise velocities from the Maxwellian at `T_bath`; when `False` use the default uniform IC |

### `info` dictionary

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mass` | float | — | Translational mass m |
| `inertia` | float | — | Moment of inertia I |
| `length` | float | — | Rod half-length L |
| `ev` | float | 1.0 | Translational restitution coefficient (1 = elastic) |
| `om` | float | 1.0 | Rotational restitution coefficient (1 = elastic) |
| `cutoff` | float | 0.1 | Angular cutoff (rad) for near-parallel pair fallback |
| `cross_section` | str | `"maxwell"` | `"maxwell"` or `"hard_needle"` |
| `initial_angle_amplitude` | float | — | Perturbation amplitude A for `perturbed_uniform_angle` IC |
| `initial_angle_shift` | float | — | Phase shift for the perturbation |
| `initial_angle_wavelength` | int | — | Wavenumber k of the perturbation |

---

## 11. Test index

All tests live in `tests/cfmz/`.  Standard parameters (unless noted):
nlocal = 10⁶, bins = 256, dt = 0.05, nsteps = 1000, seed = 47.

### Maxwell kernel (tests 0–12)

| Test | IC | ν | Vlasov force F(θ) | Transport | Notes |
|------|-----|---|-------------------|-----------|-------|
| `test_0` | uniform | 10 | — | off | Pure collision, no transport |
| `test_1` | uniform | 10 | — | on | Baseline: collision + transport |
| `test_2` | uniform | 4 | −(θ−θ_av) | on | Quadratic mean-field, α=1 |
| `test_3` | uniform | 4 | −4(θ−θ_av) | on | Quadratic mean-field, α=4 |
| `test_4` | uniform | 4 | −sin(θ−θ_av) | on | Kuramoto mean-field |
| `test_5` | perturbed | 4 | −sin(θ−θ_av) | on | Kuramoto, perturbed IC |
| `test_6` | uniform | 4 | −sin(θ−θ_av) | on | Kuramoto, seed=49 |
| `test_7` | uniform | 20 | −sin(θ−θ_av) | on | Kuramoto, high ν |
| `test_8` | uniform | 4 | Onsager | on | |
| `test_9` | uniform | 0.5 | Onsager | on | Low collision rate |
| `test_10` | uniform | 20 | Onsager | on | High collision rate |
| `test_11` | uniform | — | Onsager | on | No collisions — energy conservation check |
| `test_12` | perturbed | 4 | Onsager, L=√12 | on | Symmetry-broken IC, shows I→N transition |

### Hard-needle NTC kernel (tests 13–24)

These tests mirror tests 0–12 with the Onsager excluded-volume cross-section
W = |g·n| · L|sin(Δθ)| and Bird's NTC acceptance–rejection.  Test 11 has no
counterpart (cross-section is irrelevant with no collisions).

| Test | Mirrors | IC | ν | Vlasov force | Notes |
|------|---------|-----|---|-------------|-------|
| `test_13` | `test_1` | uniform | 10 | — | Baseline, NTC |
| `test_14` | `test_0` | uniform | 10 | — | Pure collision, NTC |
| `test_15` | `test_2` | uniform | 4 | −(θ−θ_av) | Quadratic α=1, NTC |
| `test_16` | `test_3` | uniform | 4 | −4(θ−θ_av) | Quadratic α=4, NTC |
| `test_17` | `test_4` | uniform | 4 | −sin(θ−θ_av) | Kuramoto, NTC |
| `test_18` | `test_5` | perturbed | 4 | −sin(θ−θ_av) | Kuramoto, perturbed IC, NTC |
| `test_19` | `test_6` | uniform | 4 | −sin(θ−θ_av) | Kuramoto, seed=49, NTC |
| `test_20` | `test_7` | uniform | 20 | −sin(θ−θ_av) | Kuramoto, high ν, NTC |
| `test_21` | `test_8` | uniform | 4 | Onsager | NTC |
| `test_22` | `test_9` | uniform | 0.5 | Onsager | Low ν, NTC |
| `test_23` | `test_10` | uniform | 20 | Onsager | High ν, NTC |
| `test_24` | `test_12` | perturbed | 4 | Onsager, L=√12 | I→N transition, NTC |

### Andersen thermostat — isothermal runs (tests 25–27)

These tests add the Andersen thermostat to the Onsager (test_12) setup
to replicate isothermal liquid-crystal experiments.  The critical spinodal
temperature for L = √12 is T_c = 8/π ≈ 2.55.

| Test | T_bath | α = L²/T | nu_bath | Phase | Notes |
|------|--------|----------|---------|-------|-------|
| `test_25` | 8.0 | 1.5 | 4.0 | **Isotropic** | α ≪ α_c; thermostat heats system, fluctuations dominate |
| `test_26` | 0.5 | 24 | 4.0 | **Nematic** | α ≫ α_c; Onsager ordering wins despite fixed low T |
| `test_27` | sweep | 0.2–8.0 | 4.0 | Both | nlocal=10⁵, 14 temperatures; plots σ² vs T_bath |

**test_27** runs 14 simulations at temperatures
[0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
and saves the phase diagram to `output/test_27/phase_diagram.{pdf,png}`.

---

## 12. Output format

Each test writes to `output/<prefix>_output_cfmz_<collision_type>/`:

| File | Contents |
|------|----------|
| `dsmc_<step>_*.pdf/png` | Velocity and orientation histograms |
| `dsmc_*_temperature.pdf` | Temperature vs time |
| `dsmc_*_energy.pdf` | Kinetic and (if applicable) interaction energy vs time |
| `vlasov_energy.pdf/png` | ‖V(θ)‖ vs time (Onsager tests only) |
| `history.pickle` | Python dict: `step`, `temperature`, `energy`, `interaction_energy`, `total_energy`, `momentum_1/2`, `ang_momentum`, `circular_var` |

For test_27 the phase diagram is additionally saved to
`output/test_27/phase_diagram.{pdf,png}`.

History files can be loaded with:

```python
import pickle
with open("output/test_26_output_cfmz_nanbu/history.pickle", "rb") as f:
    h = pickle.load(f)
# h["temperature"], h["circular_var"], h["total_energy"], ...
```
