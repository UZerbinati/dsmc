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
needle-like (oriented rigid rod) systems.

See **[CFMZ.md](CFMZ.md)** for full documentation: the kinetic equation,
time integration scheme, collision operators (Maxwell and hard-needle NTC),
Onsager mean-field force, isotropic–nematic phase transition, Andersen
thermostat, diagnostics, configuration reference, and the complete test index
(tests 0–27).

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
  cfmz/         test_0–27 (test_0–12: Maxwell; test_13–24: hard-needle NTC; test_25–27: Andersen thermostat)
```

## Output

Each test writes output to `output/<prefix>_output_<solver>_<collision_type>/`:

| File | Contents |
|---|---|
| `dsmc_<step>_*.pdf/png` | Spatial or velocity-space snapshots |
| `dsmc_*_temperature/energy.pdf` | Time-history plots |
| `history.pickle` | Python dict with arrays: `step`, `temperature`, `energy` (kinetic), `interaction_energy`, `total_energy`, `momentum_1/2`, … |
| `dsmc_<step>_observables.pickle` | Spatial field arrays for post-processing |
