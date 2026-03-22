# dsmc

Particle-based **Direct Simulation Monte Carlo (DSMC)** solvers for two kinetic
equations, parallelised with PETSc/MPI.

## Solvers

### `BoltzmannDSMC`
Space-inhomogeneous Boltzmann equation with Maxwell molecules.
Particles carry 2D velocity and live in a spatial domain; transport and
per-cell collisions are split with Strang splitting.

| Test | Description |
|---|---|
| `sod` | Sod shock tube on [0, 1] |
| `cylinder_flow` | 2D flow past a circular cylinder (periodic in x, specular walls) |

### `CFMDSMC`
Homogeneous CFM (Carrillo–Favre–Marchetti) kinetic equation for oriented rigid
rods.  Each particle carries translational velocity **v**, orientation
θ ∈ [0, 2π), and angular velocity ω.  An optional mean-field (Vlasov) force
acts on ω.

| Test | Description |
|---|---|
| `uniform_angle` | Angles drawn uniformly from [0, 2π) |
| `perturbed_uniform_angle` | Angles drawn from 1 + A cos(kθ) |

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

# CFM tests
cd tests/cfm
./run_tests.sh
./run_tests.sh -n 2 -nlocal 1000000
```

Individual tests accept PETSc-style flags:

```bash
mpirun -n 4 python tests/boltzmann/test_0.py \
    -nlocal 1000000 -nsteps 200 -nu 100 -dt 0.001

mpirun -n 2 python tests/cfm/test_8.py \
    -nlocal 500000 -nsteps 500 -nu 10 -monitor_every 50
```

## Repository layout

```
dsmc/
  boltzmann/    BoltzmannDSMC — mesh, initialisation, transport, collision
  cfm/          CFMDSMC       — mesh, initialisation, transport, collision
  plot.py       Plotting: spatial observables, histograms, time histories
  utils.py      Shared PETSc/MPI helpers and Matplotlib style defaults
tests/
  boltzmann/    test_0–2 (Sod shock tube), test_3 (cylinder flow)
  cfm/          test_0–14
```

## Output

Each test writes output to `output/<prefix>_output_<solver>_<collision_type>/`:

| File | Contents |
|---|---|
| `dsmc_<step>_*.pdf/png` | Spatial or velocity-space snapshots |
| `dsmc_*_temperature/energy.pdf` | Time-history plots |
| `history.pickle` | Python dict with arrays: `step`, `temperature`, `energy`, `momentum_1/2`, … |
| `dsmc_<step>_observables.pickle` | Spatial field arrays for post-processing |
