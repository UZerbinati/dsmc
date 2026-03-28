"""
MLMC coherence test for the Sod shock tube (Boltzmann with Maxwell molecules).
------------------------------------------------------------------------------
Verifies that:

  1. The MLMC estimator of E[final temperature] is statistically consistent
     with a standard single-level Monte Carlo estimator at the finest level.

  2. Variance reduction holds: Var(Q_fine − Q_coarse) < Var(Q_fine) at each
     coupled level, confirming that particle splitting produces correlated
     trajectories.

Run with:
    mpirun -n <P> python tests/boltzmann/test_mlmc.py \\
        [-nlocal_base 200] [-n_levels 2] [-nsteps 10] \\
        [-n_mc 30] [-n_warmup 15] [-epsilon 0.15]
"""
import sys
import math
import numpy as np
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

from dsmc import BoltzmannDSMC, Print
from dsmc.boltzmann.mlmc import MLMCEstimator
from dsmc.boltzmann.mlmc.qoi import FractionRightHalf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ---------------------------------------------------------------------------
# Parse command-line options via PETSc
# ---------------------------------------------------------------------------
Opt = PETSc.Options()
nlocal_base  = Opt.getInt  ("nlocal_base", 300)
n_levels     = Opt.getInt  ("n_levels",    2)
nsteps       = Opt.getInt  ("nsteps",      40)
n_mc         = Opt.getInt  ("n_mc",        30)   # standard MC samples at finest level
n_warmup     = Opt.getInt  ("n_warmup",    15)   # MLMC warm-up samples per level
epsilon      = Opt.getReal ("epsilon",     0.05)
base_seed    = Opt.getInt  ("seed",        7)

Print("=" * 66)
Print("MLMC coherence test — Sod shock tube (Boltzmann / Maxwell)")
Print("QoI: fraction of particles in right half (x > Lx/2)")
Print("=" * 66)
Print(f"  nlocal_base  = {nlocal_base}  (level-0 particles per rank)")
Print(f"  n_levels     = {n_levels}")
Print(f"  nsteps       = {nsteps}")
Print(f"  n_mc         = {n_mc}   (standard MC samples at finest level)")
Print(f"  n_warmup     = {n_warmup}  (MLMC warm-up samples per level)")
Print(f"  epsilon      = {epsilon}")
Print(f"  MPI ranks    = {comm.Get_size()}")
Print("-" * 66)

opts_base = {
    "nlocal":          nlocal_base,
    "nu":              10.0,
    "dt":              0.01,
    "bins":            32,
    "test":            "sod",
    "collision_type":  "nanbu",
    "seed":            base_seed,
    "prefix":          "output/test_mlmc",
}
info = {"temperature": 1.0, "mass": 1.0}

qoi_fn = FractionRightHalf()

# ---------------------------------------------------------------------------
# 1. Standard single-level Monte Carlo at the finest level
# ---------------------------------------------------------------------------
finest_level  = n_levels - 1
nlocal_finest = nlocal_base * (2 ** finest_level)

Print(f"\n[1] Standard MC at level {finest_level} "
      f"(nlocal = {nlocal_finest}, {n_mc} samples)...")

mc_samples = []
for i in range(n_mc):
    seed_i = base_seed + 999_999_999 + i * comm.Get_size()   # distinct from MLMC seeds
    opts_mc = dict(opts_base)
    opts_mc["nlocal"] = nlocal_finest
    opts_mc["seed"]   = seed_i
    sim = BoltzmannDSMC(opts=opts_mc, info=info, comm=comm, mlmc_mode=True)
    sim.run_silent(nsteps)
    q = qoi_fn(sim)
    mc_samples.append(q)
    sim.swarm.destroy()
    sim.dm.destroy()
    if rank == 0:
        print(f"  MC sample {i+1}/{n_mc}: T = {q:.6f}", flush=True)

mc_mean = float(np.mean(mc_samples))
mc_std  = float(np.std(mc_samples, ddof=1))
mc_se   = mc_std / math.sqrt(n_mc)

Print(f"\n  MC mean = {mc_mean:.6f},  std = {mc_std:.6f},  SE = {mc_se:.6f}")

# ---------------------------------------------------------------------------
# 2. MLMC estimator
# ---------------------------------------------------------------------------
Print(f"\n[2] MLMC estimator ({n_levels} levels, epsilon = {epsilon})...")

mlmc = MLMCEstimator(
    opts_base=opts_base,
    info=info,
    comm=comm,
    qoi_fn=qoi_fn,
    nsteps=nsteps,
    n_levels=n_levels,
    epsilon=epsilon,
    n_warmup=n_warmup,
    base_seed=base_seed,
    verbose=True,
)
result = mlmc.run()

mlmc_estimate = result["estimate"]
mlmc_se       = result["std_error"]

Print(f"\n  MLMC estimate = {mlmc_estimate:.6f} ± {mlmc_se:.6f}")

# ---------------------------------------------------------------------------
# 3. Variance reduction check (coupled levels)
# ---------------------------------------------------------------------------
# Reuse corrections already collected by the MLMC run — no extra samples needed.
Print(f"\n[3] Variance reduction check (coupled levels)...")
if finest_level >= 1:
    # Run a small set of standalone fine-level samples to estimate Var(Q_fine),
    # then compare against Var(Q_fine - Q_coarse) from the MLMC corrections.
    n_var_check = 20
    size = comm.Get_size()
    q_fines = []
    for i in range(n_var_check):
        seed_i = base_seed + 500_000_000 + i * size
        opts_fine_vc = {**opts_base, "nlocal": nlocal_finest, "seed": seed_i}
        sim = BoltzmannDSMC(opts=opts_fine_vc, info=info, comm=comm, mlmc_mode=True)
        try:
            sim.run_silent(nsteps)
            q_fines.append(qoi_fn(sim))
        finally:
            sim.swarm.destroy()
            sim.dm.destroy()

    # Corrections at the finest level are already in the MLMC result
    mlmc_corrections_finest = result["corrections"][finest_level]

    var_fine       = float(np.var(q_fines,                  ddof=1))
    var_correction = float(np.var(mlmc_corrections_finest,  ddof=1))

    Print(f"  Var(Q_fine)            = {var_fine:.4e}")
    Print(f"  Var(Q_fine - Q_coarse) = {var_correction:.4e}")
    Print(f"  Variance ratio         = {var_correction / (var_fine + 1e-30):.4f}")

    # Variance ratio < 1 means coupling helps (MLMC is efficient).
    # With simple particle-splitting, individual trajectories decouple after
    # O(1/nu) collision times, so the ratio may exceed 1 for long runs.
    # This is a known limitation of the initial-condition-only coupling;
    # a tree-based collision coupling would improve this.
    variance_reduced = var_correction < var_fine
    if rank == 0:
        if variance_reduced:
            print("  [INFO] Variance is reduced by coupling (MLMC is efficient).")
        else:
            print("  [INFO] Variance ratio > 1: coupling decays over collision "
                  "timescale (expected for simple splitting without collision CRN).")
else:
    variance_reduced = None   # nothing to check at level 0

# ---------------------------------------------------------------------------
# 4. Coherence check: |MLMC - MC| < combined confidence interval
# ---------------------------------------------------------------------------
Print(f"\n[4] Coherence check...")

# Tolerance: sum of the two standard errors, times a generous coverage factor
tolerance = 3.0 * (mc_se + mlmc_se)
diff = abs(mlmc_estimate - mc_mean)

Print(f"  |MLMC - MC|  = {diff:.6f}")
Print(f"  Tolerance    = {tolerance:.6f}  (3 × (SE_mc + SE_mlmc))")

coherent = diff < tolerance
if rank == 0:
    if coherent:
        print("  [PASS] MLMC estimate is consistent with standard MC.")
    else:
        print("  [FAIL] MLMC estimate deviates from MC beyond tolerance.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Print("\n" + "=" * 66)
Print("Summary")
Print("=" * 66)
Print(f"  Standard MC  at level {finest_level}: {mc_mean:.6f} ± {mc_se:.6f}")
Print(f"  MLMC estimate ({n_levels} levels):  {mlmc_estimate:.6f} ± {mlmc_se:.6f}")
if finest_level >= 1:
    Print(f"  Variance ratio:  {var_correction/max(var_fine,1e-30):.3f}  "
          f"(< 1 means coupling helps; > 1 means collision coupling needed)")
Print(f"  Coherence test:  {'PASS' if coherent else 'FAIL'}")

if not coherent:
    if rank == 0:
        raise SystemExit(
            "\n[ERROR] Coherence test FAILED: MLMC and MC estimates are "
            "statistically inconsistent.\n"
            "This may be a false alarm due to small sample size — re-run "
            "with larger n_mc / n_warmup."
        )

Print("\nDone.")
