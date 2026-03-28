"""
Coupled (fine, coarse) simulation pair for one MLMC sample.
"""
import numpy as np
from dsmc.boltzmann import BoltzmannDSMC

# Large constant added to the seed of the coarse simulation so that its RNG
# stream is independent from—but deterministically related to—the fine sim.
_COARSE_SEED_OFFSET = 1_000_000


def _split_state(fine_state: dict) -> dict:
    """Produce a coarse initial state by even-index decimation of fine particles.

    Every other particle from the fine state is kept on this rank.  Particle
    weights are doubled so that the physical number density is preserved.

    Parameters
    ----------
    fine_state : dict
        Snapshot returned by :meth:`BoltzmannDSMC.get_state`.

    Returns
    -------
    dict
        Coarse state in the same format as :meth:`BoltzmannDSMC.get_state`,
        with ``rng_state`` set to ``None`` (the coarse sim uses its own seed).
    """
    idx = np.arange(0, fine_state["nlocal"], 2)
    return {
        "coords":   fine_state["coords"][idx].copy(),
        "velocity": fine_state["velocity"][idx].copy(),
        "weight":   fine_state["weight"][idx].copy() * 2.0,
        "nlocal":   int(len(idx)),
        "rng_state": None,
    }


class CoupledLevelPair:
    """Manages one (fine, coarse) coupled simulation pair for MLMC level ℓ.

    At level ℓ the fine simulation has ``nlocal_fine = nlocal_base * 2**level``
    particles per rank and the coarse has ``nlocal_coarse = nlocal_fine // 2``.
    The two simulations share the same initial particle positions and velocities
    (via particle splitting from the fine initial condition), which correlates
    their trajectories and reduces ``Var(Q_fine − Q_coarse)``.

    Parameters
    ----------
    opts_base : dict
        BoltzmannDSMC options at the base level (level 0).  ``nlocal`` in this
        dict is the base count; the fine and coarse counts are derived from
        ``level``.
    level : int
        MLMC level index (≥ 1).
    info : dict
        Physical parameters passed to BoltzmannDSMC.
    comm : MPI communicator
    qoi_fn : callable(sim) -> float
        Extracts a scalar QoI from a finished simulation.  Must use allreduce
        so all ranks return the same value.
    nsteps : int
        Number of time steps per simulation run.
    """

    def __init__(self, opts_base, level, info, comm, qoi_fn, nsteps):
        if level < 1:
            raise ValueError("CoupledLevelPair requires level >= 1")
        self.level = level
        self.info = dict(info)
        self.comm = comm
        self.qoi_fn = qoi_fn
        self.nsteps = nsteps

        nlocal_fine = int(opts_base["nlocal"]) * (2 ** level)
        nlocal_coarse = nlocal_fine // 2

        self.opts_fine = dict(opts_base)
        self.opts_fine["nlocal"] = nlocal_fine

        self.opts_coarse = dict(opts_base)
        self.opts_coarse["nlocal"] = nlocal_coarse

    def run_sample(self, seed: int) -> tuple:
        """Run one coupled (fine, coarse) pair and return (Q_fine, Q_coarse).

        1. Instantiate and initialise the fine simulation with *seed*.
        2. Capture the fine initial state with :meth:`BoltzmannDSMC.get_state`.
        3. Instantiate the coarse simulation with *seed* + offset.
        4. Overwrite the coarse initial state with decimated fine particles.
        5. Run both simulations with :meth:`BoltzmannDSMC.run_silent`.
        6. Extract QoI values; destroy PETSc objects; return the pair.

        Parameters
        ----------
        seed : int
            Base RNG seed for the fine simulation.

        Returns
        -------
        (Q_fine, Q_coarse) : tuple of float
        """
        # --- Fine simulation ---
        opts_fine = dict(self.opts_fine)
        opts_fine["seed"] = seed
        fine_sim = BoltzmannDSMC(
            opts=opts_fine, info=self.info, comm=self.comm, mlmc_mode=True
        )
        fine_state = fine_sim.get_state()

        # --- Coarse simulation ---
        opts_coarse = dict(self.opts_coarse)
        opts_coarse["seed"] = seed + _COARSE_SEED_OFFSET
        coarse_sim = BoltzmannDSMC(
            opts=opts_coarse, info=self.info, comm=self.comm, mlmc_mode=True
        )
        coarse_state = _split_state(fine_state)
        coarse_sim.set_state(coarse_state)

        # --- Run both ---
        fine_sim.run_silent(self.nsteps)
        coarse_sim.run_silent(self.nsteps)

        # --- Extract QoI ---
        Q_fine = self.qoi_fn(fine_sim)
        Q_coarse = self.qoi_fn(coarse_sim)

        # --- Destroy PETSc objects (no Python GC for C objects) ---
        fine_sim.swarm.destroy()
        fine_sim.dm.destroy()
        coarse_sim.swarm.destroy()
        coarse_sim.dm.destroy()

        return Q_fine, Q_coarse
