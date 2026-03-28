"""
Quantities of interest (QoI) for MLMC with BoltzmannDSMC.

Each QoI is a callable that accepts a finished BoltzmannDSMC instance and
returns a scalar float **identical on all MPI ranks** (achieved via
allreduce).
"""
import numpy as np
from mpi4py import MPI


class FinalTemperature:
    """Mean kinetic temperature at the final time step.

    T = (2/d) · E_kin / (N · m)

    where the sum is over all particles globally (allreduced).
    """

    def __call__(self, sim) -> float:
        vel = sim.swarm.getField("velocity")
        V = vel.reshape(sim.nlocal, sim.dim)
        local_energy = 0.5 * sim.mass * float(np.sum(V * V))
        local_n = sim.nlocal
        sim.swarm.restoreField("velocity")
        global_energy = sim.comm.allreduce(local_energy, op=MPI.SUM)
        global_n = sim.comm.allreduce(local_n, op=MPI.SUM)
        return (2.0 / sim.dim) * global_energy / (global_n * sim.mass)


class FinalEnergy:
    """Total kinetic energy at the final time step (allreduced)."""

    def __call__(self, sim) -> float:
        vel = sim.swarm.getField("velocity")
        V = vel.reshape(sim.nlocal, sim.dim)
        local_energy = 0.5 * sim.mass * float(np.sum(V * V))
        sim.swarm.restoreField("velocity")
        return sim.comm.allreduce(local_energy, op=MPI.SUM)


class MeanXVelocity:
    """Mean x-component of velocity at the final time step.

    Unlike temperature, this quantity is NOT exactly conserved by the
    Sod shock tube dynamics (boundary reflections create a net x-momentum
    signal), so it has nonzero variance across samples — making it a useful
    QoI for variance-reduction tests.
    """

    def __call__(self, sim) -> float:
        vel = sim.swarm.getField("velocity")
        V = vel.reshape(sim.nlocal, sim.dim)
        local_sum_vx = float(np.sum(V[:, 0]))
        local_n = sim.nlocal
        sim.swarm.restoreField("velocity")
        global_sum_vx = sim.comm.allreduce(local_sum_vx, op=MPI.SUM)
        global_n = sim.comm.allreduce(local_n, op=MPI.SUM)
        return global_sum_vx / global_n


class FractionRightHalf:
    """Fraction of particles in the right half of the Sod domain (x > Lx/2).

    Starts at ~1/9 (density ratio 1:8) and evolves as the Sod shock develops.
    Has statistical noise ∝ 1/√N and genuine dynamics — good for variance tests.
    """

    def __call__(self, sim) -> float:
        Lx = sim.info.get("Lx", 1.0)
        celldm = sim.swarm.getCellDMActive()
        coord_names = celldm.getCoordinateFields()
        pos = sim.swarm.getField(coord_names[0])
        X = pos.reshape(sim.nlocal, sim.mesh_dim)
        local_right = int(np.sum(X[:, 0] > 0.5 * Lx))
        local_n = sim.nlocal
        sim.swarm.restoreField(coord_names[0])
        global_right = sim.comm.allreduce(local_right, op=MPI.SUM)
        global_n = sim.comm.allreduce(local_n, op=MPI.SUM)
        return global_right / global_n


class TimeAveragedTemperature:
    """Time-averaged temperature from ``sim.history``.

    Requires that :meth:`BoltzmannDSMC.run_silent` (or ``run``) has been
    called and populated ``sim.history["temperature"]``.  The average is
    taken over all recorded steps (including step 0).
    """

    def __call__(self, sim) -> float:
        temps = sim.history["temperature"]
        if not temps:
            raise RuntimeError("history is empty; did you call run_silent first?")
        return float(np.mean(temps))
