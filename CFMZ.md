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
| `collision_kind` | str | `"boltzmann"` | Inhomogeneous-solver collision kernel: `"boltzmann"` or `"enskog"` (Parsons-Lee). See §14. |
| `particle_weight` | float | `1.0` | F_N = N_phys/N_sim, decoupling sample count from physical density. Read by Enskog kernel and VTK density/cell_eta. See §14.11. |

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

---

## 13. Discotic extension (`CFMZDiscDSMCHomo`)

The CFMZ family is extended to **discotic** liquid crystals — 2-D
disc-shaped (oblate-coin) particles whose orientation θ is the in-plane
angle of a labelled axis on the disc.  Discotic particles are
effectively 4-fold symmetric (rotation by π/2 returns an equivalent
configuration), so their critical orientational order parameter is

```
R₄ = |⟨e^{4iθ}⟩|
```

rather than the rod's R₂.  Three things change relative to the
calamitic (needle) solver: the **order-parameter family**, the
**mean-field interaction kernel**, and — most consequentially — the
**collision contact mechanics**.

### 13.1 Order-parameter family

The diagnostics now expose a generalised harmonic family

```
R_n = |⟨e^{i n θ}⟩|,   n ∈ {1, 2, 4, 6}
```

via the new opt ``opts["n_modes"]`` (default ``[2]`` to preserve the
calamitic behaviour; ``CFMZDiscDSMCHomo`` overrides this default to
``[1, 2, 4, 6]``).  Each requested harmonic gets its own history key
``circular_var_n{n}``, and ``plot_history`` overlays them on a single
``<prefix>_variance_modes`` figure.  The legacy
``history["circular_var"]`` key is preserved untouched and continues
to track whichever harmonic is selected by ``opts["variance"]``
(``"circle"`` ⇒ R₁, ``"real_projective_plane"`` ⇒ R₂).

For a discotic system one expects:

| Phase | Signature |
|---|---|
| Isotropic | All R_n ≈ 0 |
| Tetratic (4-fold orientational) | R₄ ≫ 0; R₁, R₂, R₆ ≈ 0 |
| Hexatic (6-fold orientational) | R₆ ≫ 0 (special parameter regimes) |

### 13.2 Onsager pair potential and Vlasov torque

The 4-fold-symmetric Onsager-type potential is

```
W_disc(θ₁, θ₂) = |sin(2(θ₁ − θ₂))|
```

with minima at Δθ ∈ {0, π/2, π, 3π/2} (all equivalent under disc
4-fold symmetry) and maxima at the 45° configurations.  The Vlasov
torque is the discrete gradient of the CIC convolution V_eff = W_disc ∗ ρ,
identical in form to the rod case (§5) — only the kernel matrix
changes from ``|sin(Δθ)|`` to ``|sin(2 Δθ)|``.

Linear stability of the uniform isotropic state ρ₀ = 1/(2π) gives the
Fourier expansion

```
|sin 2x| = 2/π − (4/π) Σ_{m≥1} cos(4mx) / (4m²−1)
```

The unstable mode is **cos(4θ)** (rather than the calamitic cos(2θ));
its dominant Fourier coefficient `−4/(3π)` matches the rod case so the
spinodal is still at α_c = 3π/2 — but now α multiplies the discotic
kernel and the order parameter that goes critical is R₄.

The class auto-builds the ``vlasov_force`` and ``interaction_energy``
callables from W_disc using the same CIC θ-grid machinery as
``test_12.py`` (`_disc_onsager_factory` in ``dsmc/cfmz/disc.py``).
Pass your own callables to override.

### 13.3 Collision rule for discs — why and how it changes

The needle collision kernel in ``dsmc/cfmz/collision.py`` treats each
particle as a rigid **line of length 2L**.  The contact arm
``r_i = ℓ ν_i`` with ``ℓ ∈ [0, L]`` is sampled along the rod axis,
the contact-arm tip on rod j is fixed at ``r_j = L ν_j``, and the
impulse denominator carries a moment-of-inertia term
``(c_i² + c_j²)/I`` with ``c_k = r_k × n``.  Near-parallel pairs
(|sin Δθ| ≈ 0) make this denominator near-singular and trigger a
spherical-fallback branch.

For a 2-D disc of radius R the geometry is fundamentally different:
contact happens at the rim, on the line of centres.  The contact arm
is no longer along the orientation axis but along the contact normal
itself.  The lever arms vanish identically and the singular fallback
disappears.

**Hard-disc impulse derivation (`cross_section="hard_disc"`).**  Take
two identical hard discs of radius R, mass m, moment of inertia
I = m R²/2 (thin uniform disc).  At contact, their centres are 2R
apart along the contact normal **n**.  Define the contact arms

```
r_i = -R n     (centre of i to contact point)
r_j = +R n     (centre of j to contact point)
```

In 2-D the perpendicular operator sends `(x, y) → (-y, x)`, so
`r_i⊥ = -R(-n_y, n_x)` and `r_j⊥ = +R(-n_y, n_x)`.  The relative
contact velocity is

```
V = (v_i − v_j) + ω_i r_i⊥ − ω_j r_j⊥
  = (v_i − v_j) − R(ω_i + ω_j)(-n_y, n_x).
```

The lever arms vanish identically:

```
c_i = r_i × n = -R(n_x n_y − n_y n_x) = 0,
c_j = +R(n_x n_y − n_y n_x) = 0.
```

Therefore `V · n = (v_i − v_j) · n` — the relative-translational
component along **n**, exactly as in spherical hard-disc dynamics.
The impulse simplifies to

```
J = -(1 + e_v)(V · n) / (2/m)         [denom has no 1/I term]
```

and the post-collision update becomes

```
v_i' = v_i + (J/m) n,
v_j' = v_j − (J/m) n,
ω_i' = ω_i,                            [no change]
ω_j' = ω_j.                            [no change]
```

**Two key consequences** for the kinetic theory:

- **Angular momenta are conserved by collisions** for hard discs.
  ω cannot equilibrate with v through the collision operator; it can
  only be driven by the mean-field Vlasov torque (W_disc) and, if
  active, the Andersen thermostat.  In an NVE run with no Vlasov
  force the ω-distribution is a *frozen* invariant of the dynamics —
  this is precisely what ``test_disc_0`` checks.
- **No `cutoff` / spherical-fallback branch is needed.**  The
  near-parallel singularity that motivated the rod's spherical
  fallback never appears, because `(c_i² + c_j²)/I = 0` identically.

**Oriented-disc kernel (`cross_section="oriented_disc"`).**  For
problems where one wants ω to couple to v through the collisions —
for instance to study ω-relaxation in NVE runs, or when modelling
rectangular / square 4-fold particles rather than perfect discs — the
solver also implements an **NTC kernel**

```
W(Ξ₁, Ξ₂) = |g·n| · S(θ₁, θ₂),   S = R |sin(2(θ₁ − θ₂))|
```

(the 4-fold analogue of the rod's `L |sin Δθ|`) with contact arms
biased along the in-plane disc-plane direction
**ν⊥** = (-sin θ, cos θ):

```
r_i = R ν_i⊥,       r_j = R ν_j⊥,
c_k = r_k × n  ≠ 0  in general.
```

The lever arms are non-zero, and the rigid-body impulse
denominator and J expression are exactly those of the rod kernel
(see §6.1).  Bird's NTC acceptance–rejection is applied with running
maximum ``self._nu_max`` updated each step, mirroring the rod
``hard_needle`` path.

### 13.4 Cross-section summary

Set via ``info["cross_section"]``:

| Value | Cross-section weight | Acceptance | Effect on ω |
|---|---|---|---|
| ``"maxwell"`` | flat (synonym for ``hard_disc`` here) | flat | none |
| ``"hard_disc"`` *(default)* | flat (geometric) | flat | **none** — angular momenta conserved by collisions |
| ``"oriented_disc"`` | ``R · |sin(2 Δθ)|`` (4-fold Onsager) | NTC w/p `S \|g·n\| / ν_max` | full rigid-body impulse, ω updated |

### 13.5 What stays the same

- ``andersen_thermostat_step`` (resamples v, ω from the Maxwellian
  at T_bath) is reused unchanged from the parent class.
- ``transport_step`` (drift) and ``vlasov_kick_step`` (kick) are
  reused unchanged.
- The Strang DKD splitting in ``run()`` is reused unchanged.

### 13.6 New options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| ``opts["n_modes"]`` | list[int] | ``[2]`` (parent), ``[1,2,4,6]`` (disc) | Harmonics to track in diagnostics.  Each ``n`` produces a ``circular_var_n{n}`` history key. |
| ``info["radius"]`` | float | ``info.get("length", 1.0)`` | Disc radius R; used for contact arms in the disc collision kernel. |
| ``info["cross_section"]`` | str | ``"hard_disc"`` (disc) | One of ``"hard_disc"``, ``"oriented_disc"``, ``"maxwell"``. |

### 13.7 Test index — discotic homogeneous

All under ``tests/cfmz/`` with ``from dsmc import CFMZDiscDSMCHomo``:

| Test | Setup | Checks |
|------|-------|--------|
| ``test_disc_0`` | No Vlasov, ``hard_disc``, no thermostat | E conserved; ω-distribution unchanged shape; R_n ≈ 0 |
| ``test_disc_1`` | Auto disc Onsager, ``hard_disc``, NVE, perturbed θ ~ cos(4θ) | R₄ grows from 0; total E conserved; R₂ stays ≈ 0 |
| ``test_disc_2`` | Auto disc Onsager + Andersen, T_bath = 8.0 (α ≪ α_c) | All R_n < 0.1; T → T_bath |
| ``test_disc_3`` | Auto disc Onsager + Andersen, T_bath = 0.5 (α ≫ α_c) | R₄ → > 0.9; R₁, R₂, R₆ < 0.1 |
| ``test_disc_4`` | T_bath sweep over 0.2–8.0; phase-diagram analog of ``test_27`` | R₄(T_bath) drops near α_c |
| ``test_disc_5`` | Same as ``test_disc_3``; sanity check on R₁ and R₆ | R₁, R₆ ≪ R₄ throughout |

### 13.8 Inhomogeneous extension (`CFMZDiscDSMC`)

The ``CFMZDiscDSMC`` class is the positional twin of
``CFMZDiscDSMCHomo``: same orientational kernel and order parameters,
but built on top of ``CFMZNeedleDSMC`` (positions on the cell DM) so
that *positional* / *columnar* / *smectic* phenomena become accessible.

The Vlasov force is auto-built from a separate
``_disc_onsager_factory_inhomo`` because the inhomogeneous transport
step expects the signature ``F(angle, X, density)``, where ``density``
is the **per-cell** orientation histogram (one row per particle, in
probability-mass form: rows sum to 1).  The convolution
``V_eff = density @ W_disc`` therefore omits the explicit Δθ factor
that the homogeneous CIC version uses (because the per-cell density
is mass, not density-of-θ).  Per-cell collisions go through
``dsmc/cfmz/collision_disc_inhomo.py``, which mirrors the homogeneous
disc kernel.

**Smectic / positional order parameter.**  The new opt
``opts["smectic_k"]`` is a list of wavevector tuples (each tuple has
length ``spatial_dim``).  For each wavevector ``k`` the diagnostic
records

```
ψ_S(k) = |⟨exp(i k · x)⟩|
```

and stores ``smectic_re_{idx}``, ``smectic_im_{idx}``,
``smectic_abs_{idx}`` in ``history``.  ``plot_history`` overlays all
``smectic_abs_*`` curves on a single ``_smectic`` figure.

For a uniform spatial IC and no position-orientation coupling in the
dynamics, ψ_S(k) sits at the finite-N noise floor ``≈ 1/√N``.  Genuine
columnar / smectic ordering requires either a non-uniform IC, a
``translational_force`` callable that couples ω/θ to **v** (e.g. the
``"oriented_disc"`` collision mode), or boundary forcing.

| Test | Setup | Checks |
|------|-------|--------|
| ``test_disc_6`` | Inhomogeneous solver (1-D periodic, ``uniform_1d``); auto disc Onsager; Andersen thermostat at T_bath = 0.5; smectic_k at k₀ = 2π/Lx and k₁ = 4π/Lx | R₂ rises into the discotic-nematic plateau; ψ_S(k₀), ψ_S(k₁) both at noise floor 1/√N (uniform-density expectation) |

### 13.9 Switch of default kernel: from tetratic to discotic-nematic

Earlier versions of `CFMZDiscDSMCHomo` / `CFMZDiscDSMC` shipped with a
4-fold-symmetric kernel `W = |sin(2 Δθ)|` as the default — modelling
2-D coin-like particles with a labelled internal axis (effectively
2-D *squares*).  That choice produced a tetratic phase (R₄-driven).
The *physically correct* setting for a 3-D **discotic LC simulated in
a 2-D domain** uses the calamitic-form kernel `W = |sin(Δθ)|`, with
θ the in-plane projection of the disc normal (head-tail symmetric,
θ ≡ θ + π).  The Onsager second virial for thin platelets in 3-D has
exactly the same `|sin γ|` angular form as for thin rods (Eppenga &
Frenkel 1984; Veerman & Frenkel 1992); only the geometric prefactor
differs.  The unstable mode is `cos(2θ)` and the I-N transition
produces the **discotic nematic phase $N_D$** with R₂ critical.

Both kernels are now selectable via a new opt:

```python
opts["n_fold"] = 2   # default — 3-D-disc-in-2-D, R₂ critical (N_D phase)
opts["n_fold"] = 4   # tetratic / 2-D-square, R₄ critical
```

The only kernel-dependent part of the orientational machinery is the
Onsager potential (`_disc_onsager_factory`) and the `oriented_disc`
NTC cross-section (which uses `|sin(m Δθ)|` with `m = n_fold/2`).
**The hard-disc collision rule itself does not change with `n_fold`** —
the contact arms `r_i = -R n`, `r_j = +R n` are derived purely from
disc geometry, lever arms `c_k = r_k × n` vanish identically, and ω
is conserved by collisions regardless of the orientational kernel.

`n_fold` must be an even integer ≥ 2 (the kernel is built from
`|sin(m Δθ)|` and the head-tail symmetry forces an even-fold
kernel).  Other integer choices (e.g. n_fold = 6 for a hexagonal
analogue) are admissible if you have a use for them.

### 13.10 Per-cell VTK export for ParaView

The inhomogeneous solver carries spatial structure that is not directly
captured by the scalar diagnostics in `history.pickle`.  To inspect
that structure visually — local nematic order patterns, density
modulations along the director, smectic stripes — use the
``export_cell_fields_vtk`` method bound on every ``CFMZNeedleDSMC`` /
``CFMZDiscDSMC`` instance:

```python
sim = CFMZNeedleDSMC(...)
sim.run(...)                                # or call mid-run
sim.export_cell_fields_vtk(
    prefix=f"{sim.output_path}/dsmc_{step}",
    smectic_k=[(2*np.pi/Lx, 0.0)],          # registered wavevectors
    time=step * sim.dt,                      # animation time stamp
)
```

The call is collective (must be invoked on every MPI rank) but only
rank 0 actually writes the file.  Output is plain ASCII XML
``.vtr`` (vtkRectilinearGrid); no third-party dependency.

Per-cell field schema:

| Field | Components | Definition |
|---|---|---|
| `density` | scalar | particles in cell / cell volume |
| `mean_velocity` | 3 | `⟨v⟩_cell`, padded `z=0` |
| `mean_orientation` | 3 | director from local Q-tensor: `(cos θ̄, sin θ̄, 0)` with `θ̄ = ½ atan2(⟨sin 2θ⟩, ⟨cos 2θ⟩)` |
| `local_R2` | scalar | `√(⟨cos 2θ⟩² + ⟨sin 2θ⟩²)` on the cell |
| `cell_eta` | scalar | local Parsons-Lee packing fraction `(π/4) ρ L²` |
| `local_psi_re_<idx>` | scalar | `⟨cos(2θ) cos(k_idx · x)⟩_cell` per registered k |
| `local_psi_im_<idx>` | scalar | `⟨cos(2θ) sin(k_idx · x)⟩_cell` per registered k |

Empty cells receive zero in every field; this avoids ParaView
visualising spurious unit-length glyphs in regions with no particles.

ParaView pipeline for a smectic / discotic-nematic snapshot:

1. ``paraview output/<prefix>_*.vtr`` (loads the time series).
2. *Glyph* filter on cell-centre ``mean_orientation``, scale by
   ``local_R2``, type "2D Glyph" → "Edge".
3. Cell-colour the underlying grid by ``density`` to see smectic
   stripes; colour the glyph filter by ``local_psi_re_0`` to see the
   smectic phase pattern at the first registered wavevector.
4. Use the time slider to animate; ``time`` is stored as
   ``FieldData/TimeValue`` so ParaView orders frames temporally.

---

## 14. The Enskog correction (`collision_kind = "enskog"`)

The Onsager mean-field machinery in §7 captures the orientational
isotropic-nematic transition for hard rods very cleanly, but it cannot
on its own produce *positional* phases (smectic-A, columnar) that
dense rod systems develop above some packing fraction.  The reason is
in the Boltzmann collision integral itself: that integral assumes
collisions are local in space *and* uses an uncorrelated pair
distribution at contact.  Both assumptions break in dense fluids; the
**Enskog** correction restores them.

### 14.1 Why Boltzmann misses the smectic phase

Boltzmann's collision integral

```
Q_B[f, f](x, ξ) = ∫∫ |g_rel · n̂| · σ(ξ_1, ξ_2)
                    [f(x, ξ_1') f(x, ξ_2')  −  f(x, ξ_1) f(x, ξ_2)] dn̂ dξ_2
```

assumes:
1. **Locality**: both particles in a pair share the same point ``x``.
   The contact vector $\boldsymbol{\sigma}_{ij}$ between centres is
   collapsed to zero, throwing away the spatial structure that the
   contact distance would otherwise carry.
2. **Stoßzahlansatz**: ``f^{(2)}(x, x; ξ_1, ξ_2) = f(x, ξ_1) f(x, ξ_2)``.
   Equivalent to ``g(σ⁺) ≡ 1`` — particles are uncorrelated at
   contact.

Both approximations are exact in the dilute limit (low packing
fraction) but break down progressively at higher density, where
finite-size effects bias both the collision rate (more frequent in
dense regions) and the spatial structure of the post-collisional
state.  Without these effects there is no mechanism for a uniform
density to spontaneously break translational symmetry into smectic
layers — and indeed Boltzmann DSMC of hard rods at any density never
produces a smectic phase.

The classical microscopic numerical reference for the smectic-A in
dense hard rods is Bates & Frenkel (*J. Chem. Phys.* 109:6193, 1998)
in 2-D and Bolhuis & Frenkel (*Phys. Rev. E* 56:5495, 1997) in 3-D —
both via molecular dynamics, which inherently captures Enskog physics.

### 14.2 The Enskog kinetic equation

The Enskog generalisation is

```
Q_E[f, f](x, ξ) = ∫∫ |g_rel · n̂| · σ(ξ_1, ξ_2) · g(σ⁺; ρ_local) ·
                    [f(x − σ̂, ξ_1') f(x + σ̂, ξ_2')
                                              − f(x, ξ_1) f(x − σ_full, ξ_2)] …
```

Two formal differences from Boltzmann:

1. The **arguments of f at the colliding points are offset** by the
   contact vector $\boldsymbol{\sigma}$.  In particle DSMC this means
   the post-collisional state at $\mathbf{x}_i$ depends on the
   pre-collisional state at $\mathbf{x}_j = \mathbf{x}_i + \boldsymbol{\sigma}_{ij}$.
2. The **contact value of the radial distribution function**
   $g(\sigma^+; \rho_{\rm local})$ multiplies the collision rate.  At
   high density it grows above 1 (Carnahan-Starling-like) and amplifies
   the collision rate per particle — the principal microscopic effect
   of crowding.

Both ingredients introduce position-orientation correlations and
density-dependent collision frequency that Boltzmann washes out.
Together they produce, *spontaneously*, the smectic-A phase.

### 14.3 Carnahan-Starling and Parsons-Lee

The classical 3-D hard-sphere result (Carnahan-Starling 1969) is

```
Z_CS(φ) = (1 + φ + φ² − φ³) / (1 − φ)³,
g_CS(σ⁺) = (1 − φ/2) / (1 − φ)³,
```

with $\varphi = (\pi/6) \rho \sigma^3$ the volume packing fraction.
For 2-D hard discs Henderson (1975) gives

```
g_2D(σ⁺) = (1 − 7φ/16) / (1 − φ)²,    φ = π R² ρ.
```

For **2-D hard rods** Parsons (1979) and Lee (1987/1988) showed that
a Carnahan-Starling-like factor can be rolled into Onsager's
second-virial theory without altering its angular structure; the
effective rod packing fraction is

```
η = (π/4) ρ L²
```

(2-D rod-area scaling, with $L$ the rod length used in the Onsager
cross-section), and the correction factor takes the simple form

```
g_PL(η) = (1 − η/4) / (1 − η)².
```

This is the multiplicative weight we apply to the NTC acceptance
probability:

```
w_Enskog = |V · n̂_ij| · L|sin Δθ| · g_PL(η_local).
```

Cross-checks:
- Dilute limit $\eta \to 0$: $g_{\rm PL} \to 1$, recovering Boltzmann
  DSMC.
- Onset of nematic-smectic regime $\eta \approx 0.4$:
  $g_{\rm PL} \approx 1.7$, a 70 % collision-rate enhancement.
- Crystallisation regime $\eta \to 1$: $g_{\rm PL}$ diverges; we
  cap at $\eta_{\rm cap} = 0.85$ to avoid blow-up.

### 14.4 The Enskog DSMC algorithm

Implemented in `dsmc/cfmz/collision_enskog_inhomo.py` as
`nanbu_collision_step_enskog_inhomo`.  Algorithm (per timestep):

1. Compute global cell IDs for all local particles
   from positions: `i_p + j_p · nx`.  Build the cell list with
   `dsmc.utils.build_cell_lists`.
2. For each cell C:
   - Build the candidate pool $P_C = C \cup \mathrm{neighbours}(C)$
     (8 neighbours in 2-D; 2 in 1-D; periodic wrap on `bcs="periodic"`).
   - Sample $M_{\rm cand} = \lfloor \tfrac{1}{2} \nu_{\max} |C|\, \Delta t\rfloor$
     candidate pairs $(i, j)$ with $i \in C$ (avoids double-counting
     cross-cell pairs) and $j \in P_C \setminus \{i\}$.
   - For each pair: compute relative position $\mathbf{r}_{ij}$ (with
     periodic-image unwrap), reject pairs at $\|\mathbf{r}_{ij}\| > L$
     (outside the contact shell), and set
     $\hat{\mathbf{n}}_{ij} = \mathbf{r}_{ij} / \|\mathbf{r}_{ij}\|$.
   - Apply the standard rod contact-arm sampling
     $\mathbf{r}_i = \ell_i \boldsymbol{\nu}_i$ (uniform $\ell \in [0, L]$),
     $\mathbf{r}_j = L \boldsymbol{\nu}_j$ (tip on rod $j$).
   - Compute the NTC acceptance weight
     $w = |\mathbf{V}\cdot\hat{\mathbf{n}}_{ij}|\,L|\sin\Delta\theta|\,g_{\rm PL}(\eta_C)$
     and accept with probability $w/\nu_{\max}$.
   - For accepted pairs, apply the rigid-rod impulse using
     $\hat{\mathbf{n}}_{ij}$ as the contact normal — same impulse
     formula as Boltzmann (`collision_inhomo._nanbu_pair_kernel`
     lines 64-99).

The "i ∈ C only" restriction in step 2 is what avoids the cross-cell
double-counting that a naive cell-list scheme would suffer; each pair
is processed exactly once, when the loop visits the cell containing $i$.

### 14.5 Cell-sizing constraint and parameter scaling

Step 2 pulls collision partners only from the cell + neighbours.
For *all* contact pairs to be reachable the spatial cell width must be
at least the rod length:

```
L_x / bins  ≥  L     (and similarly L_y / bins for 2-D).
```

The `CFMZNeedleDSMC.__init__` constructor warns when this constraint is
violated.

**Physics-aware parameter scaling.**  The two physical scales that
the simulation must resolve are (i) the Onsager spinodal temperature
$T_{NI}$, set by the rod length $L$ alone via
$T_{NI} = (2/(3\pi)) L^2$; and (ii) the Parsons-Lee packing fraction
$\eta = (\pi/4)\rho L^2$, which controls whether the density
correction is *active* or *capped*.  We want $T_{NI}$ to land within
the swept $T_\text{bath}$ range AND $\eta$ to sit in the active range
$\eta \in [0.1, 0.7]$.

Use $L = \sqrt{12}$ (so $T_{NI} = 8/\pi \approx 2.55$ — the canonical
calamitic value).  Then $\eta = 0.5$ requires
$\rho L^2 \approx 0.64$, equivalently
$N_\text{total} \approx (4/\pi)\,\eta\,(L_x/L)^2$.  Choose
$L_x = n_\text{layers} \cdot L$ for an integer $n_\text{layers}$ so
the smectic wavelength fits the box commensurately:

| Parameter | Suggested value (Enskog smectic regime) | Rationale |
|---|---|---|
| $L$ | $\sqrt{12} \approx 3.46$ | $T_{NI} = 8/\pi \approx 2.55$ |
| `n_layers` | 50 | 50 rod-lengths across the box |
| $L_x$ | $50\sqrt{12} \approx 173.2$ | $L_x = n_\text{layers} \cdot L$ |
| `bins` | 8 | cell width $L_x/8 \approx 21.7 \gg L$ ✓ |
| $\eta$ target | 0.5 | Parsons-Lee active, not capped |
| $N_\text{total}$ | $\approx (4/\pi)\,0.5\,50^2 \approx 1592$ | derived from $\eta$ |
| $T_\text{bath}$ schedule | $\{5.0, 3.5, 2.5, 2.0, 1.5, 1.0, 0.5\}$ | crosses $T_{NI}$ |

This is what `test_needle_smectic_2d.py` and
`test_needle_smectic_2d_sweep.py` use as their defaults; they expose
`-n_layers`, `-eta`, and `-T_bath` as CLI flags so the user can scan
the parameter space.

### 14.6 Multi-rank caveat

Cross-rank ghost-particle migration is not implemented.  When
`size > 4` the constructor warns: collision partners on cell
boundaries that lie on a different MPI rank are silently dropped
from the candidate pool, biasing the statistics on those boundaries.
For research-grade runs we recommend single-rank or up-to-4-rank
operation; multi-rank Enskog with proper ghost migration is a
follow-up project.

The Boltzmann path
(`collision_kind = "boltzmann"`, the default) is unaffected by this
limitation since pair selection there is local to a single cell and
needs no neighbour information.

### 14.7 Dilute-limit regression check

The Enskog kernel reduces to Boltzmann DSMC in the limit
$\eta \to 0$ ($g_{\rm PL} \to 1$), with one residual difference: the
contact normal is still derived from positions rather than sampled
uniformly.  In the dilute regime the position-correlation contribution
to the cross-section integrates out and the moments of the relaxation
match Boltzmann within statistical noise.  This is checked numerically
in `tests/cfmz/test_needle_sod_dense.py` — see §14.9.

### 14.8 Smectic-vs-crystal distinguishing diagnostic

A smectic-A phase has 1-D positional order along the director and
remains liquid in the perpendicular plane.  A 2-D crystal has
positional order in *both* directions.  The CFMZ smectic
diagnostic registers a list of wavevectors via `opts["smectic_k"]`,
each contributing a `smectic_abs_<idx>` history key.  Registering
a wavevector along $\hat{n}$ AND another along $\hat{n}^\perp$
gives the operational signature:

| Phase | $\psi_S(\mathbf{k}\parallel \hat{n})$ | $\psi_S(\mathbf{k}\perp \hat{n})$ |
|---|---|---|
| Isotropic | ≈ 0 (noise) | ≈ 0 (noise) |
| Nematic | ≈ 0 (noise) | ≈ 0 (noise) |
| **Smectic-A** | **> noise** | ≈ 0 (noise) |
| 2-D crystal | > noise | > noise |

The "noise floor" is $\sim 1/\sqrt{N_{\rm global}}$; the diagnostic is
designed to detect ordering above this.  The test
`test_needle_smectic_2d` and the sweep
`test_needle_smectic_2d_sweep` both implement this exact
diagnostic — they register a fan of 5 wavevectors around the
natural smectic mode `m = n_layers` (so `k = m · 2π/L_x` matches
the rod-length-scale wavelength) along x̂ AND a parallel fan along
ŷ; the phase-diagram figure plots `max_m ψ_S(m·x̂)` vs
`max_m ψ_S(m·ŷ)` so the smectic-vs-crystal distinction is visible
at a glance.

Both tests expose a CLI flag `-vlasov`:
- `-vlasov 1` *(default)*: enable the calamitic Onsager mean-field
  (W = |sin Δθ|, identical to test_needle_12.py adapted to the
  inhomogeneous solver signature).  Provides the temperature-driven
  I-N transition at the canonical $T_{NI} = 8/\pi$; the Enskog
  kernel adds the position-orientation coupling that turns the
  nematic into a smectic.
- `-vlasov 0`: disable the mean-field; runs *pure Enskog* DSMC.
  Tests whether the Bates-Frenkel hard-rod smectic-A transfers to
  the kinetic / DSMC framework without any soft-potential help —
  this is research follow-up; an empirical question.

### 14.9 Tests for the Enskog needle suite

| Test | Setup | What it checks |
|------|-------|----------------|
| `test_needle_smectic_2d` | 2-D periodic, NVT at $T_\text{bath} = 0.30$, η ~ 0.6, VTK every 50 steps | smectic-A emergence in the 2-D box; $\psi_S(\hat{x})$ rises above noise, $\psi_S(\hat{y})$ stays at noise |
| `test_needle_smectic_2d_sweep` | T_bath sweep at 7 values from 1.5 down to 0.15 | three-phase diagram (I → N → Sm-A); per-T VTK snapshot for ParaView |
| `test_needle_sod_dense` | 1-D Sod tube at η_left ≈ 0.4, η_right ≈ 0.05; runs Boltzmann *and* Enskog | dilute-limit regression on the right; Enskog ≠ Boltzmann shock structure on the left (Frezzotti 1998 effect) |
| `test_needle_sod_orient` | 1-D, uniform ρ and T, vonMises(0, 4) ↔ vonMises(π/2, 4) θ-Riemann; Enskog | rod-specific: orientation-discontinuity diffusion via rod-rod collisions; ρ(x) and T(x) stay flat |

### 14.10 Sod-tube tests

Two 1-D Sod-tube tests exercise the inhomogeneous solver under
spatial gradients.  Note that 1-D simulation cannot represent a
true smectic phase (§14.8 explains why), so the role of these
tests is *validation* of the spatial transport + collision
machinery rather than smectic-phase identification.

`test_needle_sod_dense` is the standard high-density Sod tube run
twice — once with `collision_kind = "boltzmann"`, once with
`"enskog"` — and writes a side-by-side density profile.  In the
dilute right-half the kernels match within statistical noise
(dilute-limit regression).  In the dense left-half the Enskog
shock front is steeper / faster than the Boltzmann one because the
Parsons-Lee correction enhances the local collision rate; this
matches the Frezzotti 1998 prediction for hard-sphere gases,
extended here to thin rods.

`test_needle_sod_orient` is a *rod-specific* Riemann problem
without a calamitic-fluid analogue: density and temperature are
uniform on both sides of the box, but the orientation
distributions are sharply different — vonMises(0, 4) on the left,
vonMises(π/2, 4) on the right.  The orientational diffusion of
the discontinuity is driven entirely by rod-rod collisions
(through the `|sin Δθ|` cross-section).  The diagnostic plot
shows ⟨cos 2θ⟩(x) at three time slices, transitioning from a
sharp ±1 step at $t=0$ to a smooth profile by the final time.
ρ(x) and T(x) stay flat throughout — the orientation Riemann
does not couple to a density shock at the leading order.

### 14.11 Decoupled $N_{\rm sim}$ scaling via the particle weight

DSMC particles are samples of the kinetic distribution, not real
molecules.  Statistical noise on any global observable scales as
$1/\sqrt{N_{\rm sim}}$, while the *physics* (the Parsons-Lee
correction in particular) is set by the **physical** packing
fraction $\eta = (\pi/4)\rho_{\rm phys} L^2$.  In the smectic
tests we pick $\eta_{\rm target}$ for physics; on the original
$N_{\rm sim} = N_{\rm phys}$ convention this fixes the sample
count at
$$
N_{\rm phys} = (4/\pi)\,\eta_{\rm target}\,n_{\rm layers}^2,
$$
which is small ($\sim 1.6 \times 10^3$ at $\eta=0.5$,
$n_{\rm layers}=50$) and gives a noisy global noise floor of
$\sim 2.5\%$.

The standard DSMC remedy is a **per-particle weight**
$F_N = N_{\rm phys}/N_{\rm sim}$: each simulator particle
represents $F_N$ physical particles, so the *physical* density
consumed by the Enskog kernel is
$$
\rho_{\rm phys}^{\rm cell} = F_N \cdot \frac{n_{\rm sim}^{\rm cell}}{V_{\rm cell}}
$$
regardless of how $N_{\rm sim}$ is chosen.  Setting $F_N < 1$
oversamples (drives noise down at fixed physics); $F_N = 1$ is
the original behaviour.

**Wiring.**  A new key `info["particle_weight"]` (default `1.0`)
threads through `CFMZNeedleDSMC`, the Enskog kernel, and the VTK
exporter.  In `collision_enskog_inhomo.py` the only change is the
density used to compute $\eta_{\rm cell}$:

```python
rho_local = self.particle_weight * n_local / cell_volume
eta_local = 0.25 * np.pi * rho_local * L * L
```

The Nanbu/NTC sampling itself (candidate count, $\nu_{\max}$
update, acceptance ratio) stays a function of $n_{\rm sim}$, so
the collision frequency *per simulator particle* is unchanged —
only the density-dependent Parsons-Lee multiplier sees the
rescaled physical density.

**Worked example.**  $\eta = 0.5$, $n_{\rm layers} = 50$ →
$N_{\rm phys} \approx 1592$.  Running on 4 ranks at 2.5 M
particles per rank gives $N_{\rm sim} = 10^7$ and
$F_N \approx 1.59 \times 10^{-4}$.  The smectic global noise
floor drops from $1/\sqrt{1592} \approx 2.5\%$ to
$1/\sqrt{10^7} \approx 0.03\%$.  Per-cell counts in the VTK
output rise from $\sim 25$ to $\sim 10^4$, removing the
speckly-density appearance.

**F_N-invariant observables.**  $R_2$, $R_4$, $\psi_S(\mathbf{k})$,
temperature and the per-cell intensive averages
(`mean_velocity`, `mean_orientation`, `local_R2`, the smectic
projections) are all averages of intensive quantities over the
sample; they are F_N-independent.  Only `density` and `cell_eta`
in the VTK output are extensive and pick up the F_N factor.

**Activation in the tests.**  In `test_needle_smectic_2d.py` and
`test_needle_smectic_2d_sweep.py`, $N_{\rm sim}$ is now read from
the PETSc `-nlocal` option (per rank) and falls back to the
old $\eta$-derived count when omitted, so default behaviour is
unchanged.  Pass `-nlocal 2500000` on 4 ranks to recover the
worked example above.

**Recommended rank count.**  This change does not lift the
$\le 4$-rank Enskog recommendation (§14.6) — the cross-rank halo
exchange is still missing, and the pair-loss estimate at >4
ranks (now printed quantitatively at construction time) still
applies.  For $\sim 10^7$-particle runs, prefer 4 ranks at 2.5 M
particles each over 10 ranks at 1 M.

### 14.12 ParaView-only 2-D observables and the angular-fan smectic diagnostic

This subsection records three changes to the smectic-test machinery
that together let the user *observe* spontaneous Sm-A formation
without re-introducing a strong directional IC seed:

#### Visualisation grid `vis_bins`

The Enskog cell-sizing constraint requires `dx ≥ L` (rod length) so
neighbour-cell pair sampling reaches all contacts.  But a smectic
layer also has period $\lambda = L$, so the kernel mesh aliases the
density wave (Nyquist requires $dx \le L/2$).  The two requirements
are mutually exclusive on the same grid.

`export_cell_fields_vtk` accepts a `vis_bins` kwarg that overrides
the spatial bin count *for the VTK output only*.  The Enskog kernel
keeps using `self.bins`; the simulation state is unchanged.  The
recommended setting in `test_needle_smectic_2d.py` is

$$
\text{vis\_bins} \;=\; \max(8\,n_\text{layers},\, 4\,\text{bins})
$$

so $dx_\text{vis} = L/8$ — well below Nyquist, ParaView displays
clean stripes once Sm-A forms.

#### Per-cell field set

The VTK fields emitted by `export_cell_fields_vtk` are now:

| Field | Description |
|---|---|
| `density` | $F_N \cdot n_\text{cell} / V_\text{cell}$ — the physical density |
| `mean_velocity` | $\langle\mathbf{v}\rangle_\text{cell}$, padded to 3-D for ParaView glyphs |
| `mean_orientation` | $(\cos\bar\theta, \sin\bar\theta, 0)$ from the per-cell Q-tensor |
| `local_R2` | $\sqrt{\langle\cos 2\theta\rangle^2 + \langle\sin 2\theta\rangle^2}$ |
| `local_temperature` | $m(\langle v^2\rangle - \lvert\langle v\rangle\rvert^2)$ |
| `cell_eta` | $(\pi/4)\,\rho\,L^2$ — the Parsons-Lee argument |

The `local_psi_re_<idx>` / `local_psi_im_<idx>` fields emitted by
earlier versions have been **dropped**.  They projected the per-cell
quantity $\langle\cos(2\theta)\,e^{i\mathbf{k}\cdot\mathbf{x}}\rangle_\text{cell}$
at registered wavevectors $\mathbf{k}$, but the Nyquist argument
applies just as much there as to `density`: integrating $\cos(\mathbf{k}\cdot\mathbf{x})$
over a cell wider than the layer wavelength averages the wave to
zero.  The authoritative smectic-A diagnostic is the *global*
$\psi_S(\mathbf{k}) = \lvert\langle e^{i\mathbf{k}\cdot\mathbf{x}}\rangle\rvert$
in the `smectic_abs_<idx>` history series — computed without per-cell
binning, immune to the aliasing problem.

#### Removal of the per-step 2-D heatmaps

The legacy matplotlib function `plot_cfmz_observables` (and the
`self.plot_observables` binding on `CFMZNeedleDSMC`) has been removed
from `dsmc/plot.py` and `dsmc/cfmz/__init__.py`.  Per-step 2-D
heatmaps of $\rho(x,y)$, $\lvert\mathbf{u}(x,y)\rvert$, $T(x,y)$,
$S(x,y)$ are no longer emitted as `.pdf/.png` files for the
inhomogeneous solver; the same content is served from the VTK
output (read in ParaView).  The 1-D time-series plots
(`plot_history`: $T(t)$, $E(t)$, $R_n(t)$, $\psi_S(\mathbf{k}_\text{idx})(t)$)
are unchanged.

The Boltzmann solver's separate `plot_observables` function in
`dsmc/plot.py` is unaffected.

#### Angular-fan smectic diagnostic

With a near-isotropic IC (e.g. `info["initial_angle_amplitude"] = 0.01`)
the spontaneous director angle is random.  Wavevectors registered
only along $\hat{\mathbf{x}}$ and $\hat{\mathbf{y}}$ then miss the
spontaneous direction, and *every* `smectic_abs_<idx>` stays at the
$1/\sqrt{N_\text{sim}}$ noise floor even after Sm-A has formed.

The fix is purely on the test side: register an angular fan covering
$[0, \pi)$ at $n_\text{angle}$ directions and $|\text{m\_window}|$
magnitudes around the natural mode $m = n_\text{layers}$.  The default
in `test_needle_smectic_2d.py` is $n_\text{angle} = 12$ (15° resolution)
× 5 magnitudes = 60 wavevectors.  The diagnostics Allreduce buffer
grows by 2 × (60 − 10) = 100 floats per call — negligible.

Tail-averaged sanity report takes the angular max per magnitude,
which folds the fan into a single curve with one entry per $m$.  In
the Sm-A regime exactly one $m$ rises above noise; the corresponding
angle index reports the spontaneous director.  In a 2-D crystal,
*two* magnitudes rise (along the natural direction and a perpendicular
direction) — the same operational distinction documented in §14.8.

#### Smectic-test parameter harmonisation

Several inconsistencies between `test_needle_smectic_2d.py`,
`test_needle_smectic_2d_sweep.py`, and §14.5 of this document were
found during the §14.12 work and have been resolved by aligning the
single-T test to the documented values:

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `L` | $\sqrt{20}$ | $\sqrt{12}$ | Restores $T_{NI} = 8/\pi \approx 2.55$, consistent with the docstring and the rest of the calamitic suite. |
| `bins` | 32 | 8 | Matches §14.5 ("cell width $L_x/8 \approx 21.7 \gg L$").  Per-cell occupancy rises from $\sim 2.5$ to $\sim 40$ — the per-cell $\eta$ measurement that drives the Parsons-Lee correction is now statistically reliable. |
| `eta_target` | 0.8 | 0.5 | Parsons-Lee active and bounded ($g_\text{PL} = 3.5$); avoids the regime within 0.05 of the $\eta_\text{cap} = 0.85$ divergence. |
| `nsteps` | 3000 | 10000 | $t_\text{final} \approx 500$, roughly $2\times$ the thermal box-traversal time — sufficient for layer organisation. |

The `eta_target = 0.5, n_layers = 50` defaults correspond to
$N_\text{phys} \approx 1592$.  For visualisation runs the user should
pass `-nlocal 100000` (or larger) to lower the noise floor; the
particle-weight machinery from §14.11 keeps the Parsons-Lee $\eta$
fixed at the target regardless of $N_\text{sim}$.

### 14.13 The Sm-A failure mode of bare Enskog and the de Gennes–McMillan drive

Empirical investigation of the smectic test suite produced a definitive
**negative result**: the bare Boltzmann/Enskog kinetic kernel — even with
the position-derived contact normal of §14.4 and the Parsons–Lee correction
of §14.3 — **does not reproduce the Bates–Frenkel hard-rod smectic-A**, in
either the spontaneous-formation or the metastability sense.  Three
machinery additions were made to (a) diagnose this and (b) provide a
working alternative path for users who want kinetic Sm-A.

#### 14.13.1 Cross-section floor (`info["sin_dtheta_floor"]`)

In the deep-nematic regime where the Onsager Vlasov pins $R_2 \to 1$, the
NTC weight $S = L\,|\sin \Delta\theta|$ collapses to zero and the Enskog
kernel is starved of collisions — the position-orientation coupling can no
longer act because no pairs are being accepted.

A new info key `sin_dtheta_floor` (default `0.0`, fully backward
compatible) regularises the thin-rod cross-section:

$$
S = L \cdot \max\!\bigl(|\sin \Delta\theta|,\ s_\text{floor}\bigr).
$$

Physically this gives the rods an effective width $w \approx L \cdot
s_\text{floor}$, i.e. an aspect ratio $L/w \approx 1/s_\text{floor}$.
Onsager's thin-rod limit is recovered as $s_\text{floor} \to 0$.

The change is one line in each of `dsmc/cfmz/collision_inhomo.py` (Boltzmann
kernel) and `dsmc/cfmz/collision_enskog_inhomo.py` (Enskog kernel); the
existing near-parallel impulse branch (`info["cutoff"]`) already routes
accepted near-parallel pairs through the spherical-style impulse, so
nothing else changes.

The smectic tests now set `sin_dtheta_floor = sin(0.1) ≈ 0.0998` by
default (rod aspect ratio ≈ 10).

#### 14.13.2 Pre-formed smectic IC (`test = "smectic_2d"`)

To distinguish *kernel preserves Sm-A but does not reach it from uniform*
from *kernel actively destabilises Sm-A*, a new IC was added in
`dsmc/cfmz/initial_inhomo.py`:

$$
\rho(x, y) \;=\; \bar\rho\,\bigl[1 + A\,\cos\!\bigl(k_S\,(x - x_\text{min})\bigr)\bigr],
\qquad k_S = \frac{2\pi\, m}{L_x},
$$

with $m = $ `smectic_n_layers` (default $\mathrm{round}(L_x / L)$) and
$A = $ `smectic_amplitude` (default 0.5).  Positions are sampled by
rejection; orientations follow the same `cos(2θ)`-perturbed PDF as
`uniform_perturbed_2d`, so the director is along $\hat x$ (parallel to
$\mathbf{k}_S$ — the Sm-A geometry).  At $t=0$ the diagnostic reads
$\psi_S(m\hat x) \approx A/2$, all other registered wavevectors at noise.

The single-T smectic test exposes this via `-smectic_ic 1 -smectic_amp A`.

**Empirical result.**  At $T = 2.0$, $\eta = 0.7$, thermostatted, $10^6$
particles, $10^4$ steps with the cross-section floor active and the
Onsager Vlasov on, the IC seed $\psi_S = 0.25$ **decays exponentially to
the $1/\sqrt{N}$ noise floor by $t \approx 30$**, while $R_2$ is still
rising past 0.5.  This is faster than the nematic forms.  Conclusion: in
the bare Enskog framework Sm-A is not even *metastable* — it is actively
washed out by free streaming, because nothing in the kernel feeds back
from local density fluctuations into a positional drift.

#### 14.13.3 The de Gennes–McMillan smectic Vlasov drive

To recover Sm-A in the kinetic framework, one must add a smectic-coupled
soft potential — the kinetic translation of the de Gennes–McMillan
free-energy term $F_\text{dGM} \propto -\gamma\,|\psi|^2 S^2$, where
$\psi$ is the smectic order parameter and $S$ the nematic order parameter.

The per-particle potential, with the director fixed along $\hat x$:

$$
V_\text{sm}(\mathbf{x}, \theta) \;=\; -\,c\,\cos(2\theta)\,\cos(k_S\,x_0),
\qquad k_S = \frac{2\pi\,n_\text{layers}}{L_x} = \frac{2\pi}{L},
$$

with `c = smectic_coupling` the dimensionless prefactor.  Differentiating
gives a translational force and an additive angular torque:

$$
\mathbf{F}_\text{sm}(\mathbf{x}, \theta)
= -\,c\,k_S\,\cos(2\theta)\,\sin(k_S\,x_0)\,\hat x,
\qquad
\tau_\text{sm}(\mathbf{x}, \theta) = -\,2c\,\cos(k_S\,x_0)\,\sin(2\theta).
$$

**Reading the factors.**  $\cos(2\theta)$ gates the force by alignment:
aligned rods (along $\hat x$) feel the layer potential strongly,
perpendicular rods are immune, and an isotropic ensemble sees zero net
force — the drive automatically switches off when there is no nematic.
$\cos(k_S\, x_0)$ in $\tau_\text{sm}$ couples nematic-aligning torque to
density antinodes (positive at the layers, negative between them); this
is the microscopic counterpart of the $|\psi|^2 S^2$ free-energy
coupling.

**Wiring.**  The translational force slot was already present in
`CFMZNeedleDSMC.__init__` and `vlasov_kick_step` (`transport_inhomo.py`);
no module code was modified.  The drive is built in-test, mirroring the
`_onsager_vlasov_inhomo` factory pattern, in both
`tests/cfmz/test_needle_smectic_2d.py` and the sweep
`test_needle_smectic_2d_sweep.py`.  When both Onsager and smectic drives
are on, the angular torques compose additively; the smectic translational
force runs alone (Onsager has no positional component because it
allreduces the global $\theta$-density).

**Director assumption (v1).**  The drive fixes $\hat n_\text{dir} = \hat x$
to keep the smectic potential commensurate with the periodic box.  A
dynamic $\theta_\text{dir} = \tfrac{1}{2}\arg\langle e^{2i\theta}\rangle$
version is left as a v2 follow-up; for that case the cosine wave's argument
$k_S\,\mathbf{x}\cdot\hat n_\text{dir}$ is generically incommensurate with
$L_x$ along arbitrary directions, requiring either a multi-mode potential
or a per-particle nearest-image projection.  The cos(2θ) IC perturbation in
both smectic tests biases the director along $\hat x$ from the start, so
the v1 assumption is consistent with the test setup.

#### 14.13.4 New CLI flags on the smectic tests

Both `test_needle_smectic_2d.py` and `test_needle_smectic_2d_sweep.py`
expose:

| Flag | Default | Purpose |
|------|---------|---------|
| `-smectic_drive 0\|1` | `0` | Enable / disable the de Gennes–McMillan Vlasov drive. |
| `-smectic_coupling C` | `1.0` | Dimensionless prefactor $c$ in $V_\text{sm}$. |

In addition the single-T test exposes:

| Flag | Default | Purpose |
|------|---------|---------|
| `-smectic_ic 0\|1` | `0` | Replace the `uniform_perturbed_2d` IC with the pre-formed `smectic_2d` IC of §14.13.2 — used for stability testing. |
| `-smectic_amp A` | `0.5` | Density-wave amplitude $A$ of that IC; $\psi_S(t=0) \approx A/2$. |

Default behaviour (all flags zero) reproduces the pre-§14.13 test code
path bit-for-bit.

#### 14.13.5 Operational protocol

To investigate Sm-A in the kinetic framework:

1. **Confirm the bare-kernel failure** (already known; reproduce only if
   you mistrust the recipe):
   ```bash
   mpirun -n 1 python tests/cfmz/test_needle_smectic_2d.py \
       -nlocal 1000000 -nsteps 10000 \
       -T_bath 2.0 -nu_bath 16 -eta 0.7 \
       -smectic_ic 1 -smectic_amp 0.5
   ```
   $\psi_S$ should decay to noise within $t \sim 30$.

2. **Check Sm-A stability with the drive on**:
   ```bash
   ... -smectic_ic 1 -smectic_drive 1 -smectic_coupling 1.0
   ```
   $\psi_S$ should *stay* near $A/2$ rather than decay.  Tail report
   prints "✓ Sm-A signal" rather than the warning.

3. **Check spontaneous Sm-A formation from uniform**:
   ```bash
   ... -smectic_drive 1 -smectic_coupling 1.0
   ```
   $\psi_S$ should *rise* from noise to a finite value and one $m=50$
   wavevector should dominate while the others stay flat.

4. **Sweep across the I → N → Sm-A transitions** with the same coupling:
   ```bash
   mpirun -n 1 python tests/cfmz/test_needle_smectic_2d_sweep.py \
       -nlocal 1000000 -nsteps 10000 \
       -smectic_drive 1 -smectic_coupling 1.0
   ```
   The phase diagram should show $\max_m \psi_S(m\hat x)$ rising below a
   measurable $T_{NA}$ while $\max_m \psi_S(m\hat y)$ stays at noise — the
   smectic-vs-crystal distinguishing diagnostic of §14.8.

If $c = 1.0$ is poorly tuned, sweep `-smectic_coupling 0.3 / 1.0 / 3.0 /
10.0` and pick the smallest value that gives spontaneous formation in
step 3; that is the kinetic equivalent of the $\gamma$ parameter in the
de Gennes–McMillan free energy and sets where $T_{NA}$ lands.
