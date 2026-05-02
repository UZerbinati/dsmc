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
