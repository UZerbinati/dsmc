"""
Multilevel Monte Carlo (MLMC) estimator for BoltzmannDSMC.

Usage example
-------------
::

    from mpi4py import MPI
    from dsmc.boltzmann.mlmc import MLMCEstimator
    from dsmc.boltzmann.mlmc.qoi import FinalTemperature

    opts_base = {
        "nlocal": 500,
        "nu": 10.0,
        "dt": 0.01,
        "bins": 32,
        "test": "sod",
        "collision_type": "nanbu",
        "seed": 42,
        "prefix": "mlmc_output",
    }
    info = {"temperature": 1.0, "mass": 1.0}

    estimator = MLMCEstimator(
        opts_base=opts_base,
        info=info,
        comm=MPI.COMM_WORLD,
        qoi_fn=FinalTemperature(),
        nsteps=20,
        n_levels=3,
        epsilon=0.05,
        n_warmup=10,
    )
    result = estimator.run()
    print(f"MLMC estimate: {result['estimate']:.4f} ± {result['std_error']:.4f}")
"""

import math
import numpy as np
from mpi4py import MPI

from dsmc.boltzmann import BoltzmannDSMC
from .level_pair import CoupledLevelPair
from .allocation import giles_optimal_allocation, level_cost

# Seed stride between levels so that level-ℓ seeds never collide with level-ℓ'.
_LEVEL_SEED_STRIDE = 10_000_000


class MLMCEstimator:
    """Adaptive MLMC estimator wrapping :class:`BoltzmannDSMC`.

    The estimator computes

        Ê[Q_L] = Σ_{ℓ=0}^{L}  (1/n_ℓ) Σ_{i=1}^{n_ℓ}  ΔQ_ℓ^{(i)}

    where  ΔQ_0 = Q_0  and  ΔQ_ℓ = Q_ℓ − Q_{ℓ−1}  for ℓ ≥ 1.

    **Adaptive algorithm**

    1. *Warm-up* — run ``n_warmup`` samples at every level to estimate the
       per-level correction variance V_ℓ and cost C_ℓ.
    2. *Allocation* — use Giles' formula to compute the optimal n_ℓ given
       the target RMSE ``epsilon``.
    3. *Completion* — run the remaining ``max(0, n_ℓ − n_warmup)`` samples.
    4. *Assembly* — return the MLMC estimate, its standard error, and
       per-level diagnostics.

    Parameters
    ----------
    opts_base : dict
        BoltzmannDSMC options at level 0 (``nlocal`` = N_0).
    info : dict
        Physical parameters passed to every BoltzmannDSMC instance.
    comm : MPI communicator
    qoi_fn : callable(sim) -> float
        Scalar QoI extractor; must use allreduce so all ranks agree.
    nsteps : int
        Time steps per simulation run.
    n_levels : int
        Total number of levels (L = n_levels − 1).  Level ℓ uses
        ``nlocal = N_0 · 2**ℓ`` particles per rank.
    epsilon : float
        Target root-mean-squared error for the MLMC estimator.
    n_warmup : int
        Samples per level during the warm-up phase (default 10).
    base_seed : int
        Master seed; per-level per-sample seeds are derived deterministically.
    verbose : bool
        If True and rank 0, print per-level diagnostics.
    """

    def __init__(
        self,
        opts_base: dict,
        info: dict,
        comm: MPI.Comm,
        qoi_fn,
        nsteps: int,
        n_levels: int = 3,
        epsilon: float = 0.05,
        n_warmup: int = 10,
        base_seed: int = 99,
        verbose: bool = True,
    ):
        self.opts_base = dict(opts_base)
        self.info = dict(info)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.qoi_fn = qoi_fn
        self.nsteps = nsteps
        self.L = n_levels - 1
        self.epsilon = epsilon
        self.n_warmup = n_warmup
        self.base_seed = base_seed
        self.verbose = verbose

        self.nlocal_base = int(opts_base["nlocal"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full adaptive MLMC algorithm.

        Returns
        -------
        dict with keys:

        ``estimate``
            MLMC estimate of E[Q].
        ``std_error``
            Standard error of the estimator (√Var_estimator).
        ``n_samples``
            Number of samples used per level.
        ``level_means``
            Sample mean of ΔQ_ℓ at each level.
        ``level_vars``
            Sample variance of ΔQ_ℓ at each level.
        ``total_cost``
            Total particle-steps consumed.
        """
        corrections = [[] for _ in range(self.L + 1)]

        # Phase 1: warm-up
        if self.verbose and self.rank == 0:
            print(f"[MLMC] Warm-up: {self.n_warmup} samples per level...")
        for ell in range(self.L + 1):
            samples = self._run_level(ell, self.n_warmup, sample_offset=0)
            corrections[ell].extend(samples)

        level_vars = [float(np.var(c, ddof=1)) if len(c) > 1 else float("inf")
                      for c in corrections]
        costs = [level_cost(self.nlocal_base, ell, self.nsteps, self.size)
                 for ell in range(self.L + 1)]

        if self.verbose and self.rank == 0:
            print("[MLMC] Warm-up variance estimates:")
            for ell, v in enumerate(level_vars):
                print(f"  level {ell}: Var(ΔQ) = {v:.4e},  cost = {costs[ell]:.2e}")

        # Phase 2: optimal allocation
        n_opt = giles_optimal_allocation(level_vars, costs, self.epsilon)

        if self.verbose and self.rank == 0:
            print(f"[MLMC] Optimal allocation: {n_opt}")

        # Phase 3: completion
        for ell in range(self.L + 1):
            n_extra = max(0, n_opt[ell] - self.n_warmup)
            if n_extra > 0:
                samples = self._run_level(
                    ell, n_extra, sample_offset=self.n_warmup
                )
                corrections[ell].extend(samples)

        # Phase 4: assemble
        n_samples = [len(c) for c in corrections]
        level_means = [float(np.mean(c)) for c in corrections]
        level_vars_final = [
            float(np.var(c, ddof=1)) if len(c) > 1 else 0.0
            for c in corrections
        ]

        estimate = float(sum(level_means))
        estimator_variance = sum(
            v / n for v, n in zip(level_vars_final, n_samples)
        )
        std_error = math.sqrt(max(0.0, estimator_variance))

        total_cost = sum(
            n * level_cost(self.nlocal_base, ell, self.nsteps, self.size)
            for ell, n in enumerate(n_samples)
        )

        if self.verbose and self.rank == 0:
            print(f"[MLMC] Estimate = {estimate:.6f} ± {std_error:.6f}")
            print(f"[MLMC] Samples per level: {n_samples}")
            print(f"[MLMC] Total cost (particle-steps): {total_cost:.2e}")

        return {
            "estimate":    estimate,
            "std_error":   std_error,
            "n_samples":   n_samples,
            "level_means": level_means,
            "level_vars":  level_vars_final,
            "total_cost":  total_cost,
            "corrections": corrections,  # list[list[float]], one per level
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed(self, level: int, sample_idx: int) -> int:
        """Deterministic seed: no collisions across levels or samples."""
        return self.base_seed + level * _LEVEL_SEED_STRIDE + sample_idx * self.size

    def _run_level(self, level: int, n_samples: int, sample_offset: int = 0) -> list:
        """Run *n_samples* independent corrections at *level*.

        Level 0 returns a list of Q_0 values (single standalone simulation).
        Level ℓ ≥ 1 returns a list of (Q_fine − Q_coarse) corrections from
        coupled pairs.

        All ranks participate; each element of the returned list is the same
        scalar on all ranks.
        """
        corrections = []

        if level == 0:
            for i in range(n_samples):
                seed = self._seed(0, sample_offset + i)
                q = self._run_single(seed)
                corrections.append(q)
        else:
            pair = CoupledLevelPair(
                opts_base=self.opts_base,
                level=level,
                info=self.info,
                comm=self.comm,
                qoi_fn=self.qoi_fn,
                nsteps=self.nsteps,
            )
            for i in range(n_samples):
                seed = self._seed(level, sample_offset + i)
                q_fine, q_coarse = pair.run_sample(seed)
                corrections.append(q_fine - q_coarse)

        return corrections

    def _run_single(self, seed: int) -> float:
        """Run one standalone level-0 simulation and return Q."""
        sim = BoltzmannDSMC(
            opts={**self.opts_base, "seed": seed},
            info=self.info, comm=self.comm, mlmc_mode=True,
        )
        try:
            sim.run_silent(self.nsteps)
            return self.qoi_fn(sim)
        finally:
            # PETSc objects are not garbage-collected by Python
            sim.swarm.destroy()
            sim.dm.destroy()
