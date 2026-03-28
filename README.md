# dsmc

Particle-based **Direct Simulation Monte Carlo (DSMC)** solvers for two kinetic
equations, parallelised with PETSc/MPI.

## Solvers

### `BoltzmannDSMC`
Space-inhomogeneous Boltzmann equation with Maxwell molecules.
Particles carry 2D velocity and live in a spatial domain; transport and
per-cell collisions are split with Strang splitting.

| Test file | IC / collision | Description |
|---|---|---|
| `test_0` | sod / Nanbu, ν=10 | Sod shock tube baseline |
| `test_1` | sod / Nanbu, ν=100 | Higher collision frequency |
| `test_2` | sod / Nanbu, ν=100, 100 sub-steps | Many extra collision sub-steps |
| `test_3` | sod / BGK | Sod shock tube with BGK relaxation |
| `test_4` | cylinder_flow / Nanbu | 2D flow past a circular cylinder (laminar, Re~15) |
| `test_5` | cylinder_flow / Nanbu | Vortex shedding regime (Re~150, von Kármán street) |

### `CFMZNeedleDSMC`
Homogeneous CFMZ (Carrillo–Farrell–Medaglia–Zerbinati) kinetic equation for
needle-like (oriented rigid rod) systems.  Each particle carries translational
velocity **v**, orientation θ ∈ [0, 2π), and angular velocity ω.  An optional
mean-field (Vlasov) force acts on ω.

Tests 8–12 use the **Onsager potential** W(θ₁,θ₂) = |sin(θ₁−θ₂)|, which gives
the Vlasov torque F(θ) = −L² ∫ sign(sin(θ−θ')) cos(θ−θ') ρ(θ') dθ'.  The
interaction energy E[ρ] = ∫∫ W ρ ρ dθ₁ dθ₂ and the total energy
E_kin + E[ρ] are tracked and plotted alongside the kinetic observables.

Tests 13–24 repeat tests 0–12 (where applicable) using the **hard-needle NTC
cross-section** W = |g·n| · L|sin(Δθ)| (Onsager excluded-volume kernel,
arXiv:2508.10744 Example B) instead of the Maxwell (uniform) kernel.  Bird's
No-Time-Counter acceptance–rejection method is used: `nu` is the initial
estimate of the running maximum ν_max, which adapts every step.  Test 11 has no
counterpart because the cross-section is irrelevant when collisions are
disabled; test 1 is already covered by test 13.

#### Maxwell molecules (tests 0–12)

| Test | IC | Vlasov force F(θ) | Notes |
|---|---|---|---|
| `test_0` | uniform | — | Pure collision, **no transport** |
| `test_1` | uniform | — | Baseline: collision + transport, ν=10 |
| `test_2` | uniform | −(θ−θ_av) | Quadratic, α=1 |
| `test_3` | uniform | −4(θ−θ_av) | Quadratic, α=4 |
| `test_4` | uniform | −sin(θ−θ_av) | Kuramoto meanfield |
| `test_5` | perturbed | −sin(θ−θ_av) | Kuramoto meanfield, perturbed IC |
| `test_6` | uniform | −sin(θ−θ_av) | Kuramoto meanfield, seed=49 |
| `test_7` | uniform | −sin(θ−θ_av) | Kuramoto meanfield, high ν=20 |
| `test_8` | uniform | Onsager | ν=4, bins=128 |
| `test_9` | uniform | Onsager | ν=0.5, bins=128 |
| `test_10` | uniform | Onsager | ν=20, bins=128 |
| `test_11` | uniform | Onsager | **No collisions** (pure Vlasov), bins=128 — energy conservation check |
| `test_12` | perturbed | Onsager | ν=4, bins=128, perturbed IC |

#### Hard-needle NTC cross-section (tests 13–24)

| Test | Mirrors | IC | Vlasov force F(θ) | Notes |
|---|---|---|---|---|
| `test_13` | `test_1` | uniform | — | Baseline: collision + transport, ν=10 |
| `test_14` | `test_0` | uniform | — | Pure collision, **no transport** |
| `test_15` | `test_2` | uniform | −(θ−θ_av) | Quadratic, α=1 |
| `test_16` | `test_3` | uniform | −4(θ−θ_av) | Quadratic, α=4 |
| `test_17` | `test_4` | uniform | −sin(θ−θ_av) | Kuramoto meanfield |
| `test_18` | `test_5` | perturbed | −sin(θ−θ_av) | Kuramoto meanfield, perturbed IC |
| `test_19` | `test_6` | uniform | −sin(θ−θ_av) | Kuramoto meanfield, seed=49 |
| `test_20` | `test_7` | uniform | −sin(θ−θ_av) | Kuramoto meanfield, high ν=20 |
| `test_21` | `test_8` | uniform | Onsager | ν=4, bins=128 |
| `test_22` | `test_9` | uniform | Onsager | ν=0.5, bins=128 |
| `test_23` | `test_10` | uniform | Onsager | ν=20, bins=128 |
| `test_24` | `test_12` | perturbed | Onsager | ν=4, bins=128, perturbed IC |

## Dependencies

- Python ≥ 3.10
- [PETSc](https://petsc.org/) with **petsc4py**
- **mpi4py**
- NumPy, Matplotlib

## Installation

```bash
pip install -e .
```

## Running tests

Each solver has a `run_tests.sh` in its test directory:

```bash
# Boltzmann tests
cd tests/boltzmann
./run_tests.sh                        # defaults: 5 ranks, 10 M particles/rank
./run_tests.sh -n 4 -nlocal 500000

# CFMZ needle tests
cd tests/cfmz
./run_tests.sh
./run_tests.sh -n 2 -nlocal 1000000
```

Individual tests accept PETSc-style flags:

```bash
mpirun -n 4 python tests/boltzmann/test_0.py \
    -nlocal 1000000 -nsteps 200 -nu 100 -dt 0.001

mpirun -n 2 python tests/cfmz/test_8.py \
    -nlocal 500000 -nsteps 500 -nu 10 -monitor_every 50
```

## Repository layout

```
dsmc/
  boltzmann/    BoltzmannDSMC — mesh, initialisation, transport, collision
  cfmz/         CFMZNeedleDSMC — mesh, initialisation, transport, collision
  plot.py       Plotting: spatial observables, histograms, time histories
  utils.py      Shared PETSc/MPI helpers and Matplotlib style defaults
tests/
  boltzmann/    test_0–2 (Sod shock tube, Nanbu), test_3 (Sod, BGK), test_4–5 (cylinder flow)
  cfmz/         test_0–24 (test_0–12: Maxwell; test_13–24: hard-needle NTC)
```

## Output

Each test writes output to `output/<prefix>_output_<solver>_<collision_type>/`:

| File | Contents |
|---|---|
| `dsmc_<step>_*.pdf/png` | Spatial or velocity-space snapshots |
| `dsmc_*_temperature/energy.pdf` | Time-history plots |
| `history.pickle` | Python dict with arrays: `step`, `temperature`, `energy` (kinetic), `interaction_energy`, `total_energy`, `momentum_1/2`, … |
| `dsmc_<step>_observables.pickle` | Spatial field arrays for post-processing |
