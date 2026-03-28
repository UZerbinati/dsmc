# Multilevel Monte Carlo for BoltzmannDSMC

This document describes the MLMC layer added to the `BoltzmannDSMC` solver in `dsmc/boltzmann/mlmc/`.

## Background

Standard Monte Carlo estimation of a quantity of interest Q (e.g. mean temperature, fraction of particles in a region) at the finest particle level L requires $O(\varepsilon^{-3})$ work to reach RMSE $\varepsilon$: $O(\varepsilon^{-2})$ samples each costing $O(\varepsilon^{-1})$ (since N ∝ 1/ε² for CLT noise, and cost ∝ N).

MLMC reduces this to $O(\varepsilon^{-2})$ total work (when the coupling is effective) by replacing the single fine-level estimator with a telescoping sum:

$$\hat{Q}_{MLMC} = \sum_{\ell=0}^{L} \frac{1}{n_\ell} \sum_{i=1}^{n_\ell} \Delta Q_\ell^{(i)}$$

where $\Delta Q_0 = Q_0$ and $\Delta Q_\ell = Q_\ell - Q_{\ell-1}$ for $\ell \geq 1$. Each $\Delta Q_\ell^{(i)}$ is computed from a **coupled** (fine, coarse) pair sharing the same initial conditions, so $\text{Var}(\Delta Q_\ell)$ is small and fewer samples $n_\ell$ are needed at expensive fine levels.

## Level Hierarchy

| Level ℓ | Particles per rank | Cost per sample |
|---------|-------------------|-----------------|
| 0       | N₀                | N₀ · nsteps     |
| 1       | 2N₀               | 3N₀ · nsteps    |
| ℓ       | 2^ℓ N₀            | 3 · 2^(ℓ-1) N₀ · nsteps |

All levels share the same `dt`, `nu`, `bins`, `nsteps`, and physical parameters.

## Package Structure

```
dsmc/boltzmann/mlmc/
    __init__.py      # MLMCEstimator  — adaptive algorithm + Giles allocation
    level_pair.py    # CoupledLevelPair — one (fine, coarse) coupled run
    qoi.py           # QoI callables
    allocation.py    # giles_optimal_allocation, level_cost
```

## Coupling Strategy: Particle Splitting

At level ℓ ≥ 1, the fine simulation has N_fine = N₀ · 2^ℓ particles and the coarse has N_coarse = N_fine / 2. Their initial conditions are coupled by **particle splitting**:

1. Generate N_fine particles from the RNG with seed `s`.
2. Snapshot the fine state with `BoltzmannDSMC.get_state()`.
3. Create the coarse state by **even-index decimation**: keep particles at indices 0, 2, 4, … and double their weights to preserve number density.
4. Load the coarse state into the coarse simulation with `BoltzmannDSMC.set_state()`.
5. Both simulations evolve independently from this shared starting point.

This coupling correlates the initial empirical measures of the two levels. With a longer run, individual trajectories decouple over the collision timescale O(1/ν). A tree-based collision coupling (sharing scattering angles between paired fine/coarse particles) would maintain the correlation and achieve $\text{Var}(\Delta Q_\ell) < \text{Var}(Q_\ell)$, but is not yet implemented.

## Adaptive Algorithm

`MLMCEstimator.run()` proceeds in four phases:

1. **Warm-up**: run `n_warmup` samples at each level; estimate $V_\ell = \text{Var}(\Delta Q_\ell)$ and model cost $C_\ell$.

2. **Optimal allocation** (Giles 2008):

$$n_\ell = \left\lceil \frac{2}{\varepsilon^2} \sqrt{\frac{V_\ell}{C_\ell}} \sum_{k} \sqrt{V_k C_k} \right\rceil$$

3. **Completion**: run the remaining $\max(0, n_\ell - n_\text{warmup})$ samples at each level.

4. **Assembly**: return estimate, standard error, per-level diagnostics, and raw corrections.

## API Reference

### `MLMCEstimator`

```python
from dsmc.boltzmann.mlmc import MLMCEstimator
from dsmc.boltzmann.mlmc.qoi import FractionRightHalf

estimator = MLMCEstimator(
    opts_base = {           # BoltzmannDSMC opts at level 0
        "nlocal": 500,      # particles per MPI rank at level 0
        "nu": 10.0,
        "dt": 0.01,
        "bins": 32,
        "test": "sod",
        "collision_type": "nanbu",
        "seed": 42,
        "prefix": "mlmc_out",
    },
    info      = {"temperature": 1.0, "mass": 1.0},
    comm      = MPI.COMM_WORLD,
    qoi_fn    = FractionRightHalf(),   # any QoI callable
    nsteps    = 50,
    n_levels  = 3,       # levels 0, 1, 2
    epsilon   = 0.02,    # target RMSE
    n_warmup  = 15,      # samples per level for variance estimation
    base_seed = 7,
    verbose   = True,
)
result = estimator.run()
```

**Result dict keys:**

| Key           | Type             | Description |
|---------------|------------------|-------------|
| `estimate`    | float            | MLMC estimate of E[Q] |
| `std_error`   | float            | Standard error √Var(estimator) |
| `n_samples`   | list[int]        | Samples used per level |
| `level_means` | list[float]      | Mean of ΔQ_ℓ per level |
| `level_vars`  | list[float]      | Variance of ΔQ_ℓ per level |
| `total_cost`  | float            | Total particle-steps |
| `corrections` | list[list[float]]| Raw ΔQ_ℓ samples per level |

### `BoltzmannDSMC` extensions

Three methods are added for MLMC support:

| Method | Description |
|--------|-------------|
| `get_state() -> dict` | Snapshot of all particle data and RNG state |
| `set_state(state)` | Restore snapshot (resize swarm, set coordinates + cell membership, overwrite fields) |
| `run_silent(nsteps)` | Run time loop without any file I/O or plots; populates `self.history` in memory |

The `mlmc_mode=True` constructor flag skips directory creation and matplotlib setup, which is necessary when creating many short-lived instances.

### QoI Callables (`dsmc.boltzmann.mlmc.qoi`)

Each QoI is a callable `(sim: BoltzmannDSMC) -> float` that uses `comm.Allreduce` so all MPI ranks return the same value.

| Class | Description |
|-------|-------------|
| `FinalTemperature` | Mean kinetic temperature T = (2/d) E_kin / (N m) |
| `FinalEnergy` | Total kinetic energy |
| `MeanXVelocity` | Mean x-component of velocity |
| `FractionRightHalf` | Fraction of particles with x > Lx/2 (Sod test) |
| `TimeAveragedTemperature` | Time-average of temperature from `sim.history` |

## Running the Coherence Test

```bash
source ~/Devel/fd/bin/activate
mpirun -n 4 python tests/boltzmann/test_mlmc.py \
    -nlocal_base 300 \
    -n_levels 2 \
    -nsteps 40 \
    -n_mc 30 \
    -n_warmup 15 \
    -epsilon 0.05
```

The test:
1. Runs 30 independent standard MC samples at the finest level → mean ± SE.
2. Runs the MLMC estimator → estimate ± SE.
3. Checks `|MLMC - MC| < 3 · (SE_mc + SE_mlmc)` (coherence, hard pass/fail).
4. Reports the variance ratio $\text{Var}(\Delta Q_L) / \text{Var}(Q_L)$ as an informational diagnostic.

## Known Limitations and Future Work

**Variance reduction**: The current particle-splitting coupling only correlates initial conditions. After O(1/ν) collision times, the fine and coarse trajectories decouple, so $\text{Var}(\Delta Q_\ell) \approx \text{Var}(Q_\ell)$ and the MLMC speedup is limited. A **tree-based collision coupling** — pairing coarse particle i with fine particles 2i and 2i+1 and using the same scattering angle — would maintain trajectory correlation and reduce the variance ratio below 1.

**Level-0 reuse**: Each level-0 sample creates and destroys a full `BoltzmannDSMC` instance (PETSc DM + DMSwarm + initialization). For large `n_warmup` or many level-0 samples, this could be optimised by snapshotting the initial state once and using `set_state` to reset between samples.

**Bias estimate**: `MLMCEstimator` does not currently estimate or correct for the discretisation bias at the finest level. The `epsilon` target covers only the statistical variance; the user must separately verify that the finest level is fine enough that the bias is O(ε).
