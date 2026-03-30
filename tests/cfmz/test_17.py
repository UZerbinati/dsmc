"""
Hard-needle cross-section — Kuramoto meanfield Vlasov force, uniform IC
------------------------------------------------------------------------
Same as test_4 (Kuramoto mean-field torque F(θ) = −sin(θ − θ_av), uniform IC)
but using the hard-needle NTC kernel instead of the Maxwell (uniform) kernel.

Bird's NTC acceptance–rejection method is applied:
  - Mcol_cand = floor(ν_max · N · dt / 2) candidate pairs are drawn each step.
  - Each candidate is accepted with probability |g·n| · L|sin(Δθ)| / ν_max.
  - The running maximum ν_max (stored as sim._nu_max) is updated every step.

Compare with test_4 (same parameters, Maxwell kernel) to observe the effect of
the anisotropic cross-section on Kuramoto synchronisation dynamics.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print
import numpy as np

Opt = PETSc.Options()
Print("Running homogeneous CFMZ needle DSMC — hard-needle cross-section (NTC), Kuramoto Vlasov:")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4)
nsteps = Opt.getInt("nsteps", 1000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
monitor_every = Opt.getInt("monitor_every", 100)

Print(f"  nlocal={nlocal}")
Print(f"  nu={nu}  (initial NTC estimate; adapts to max kernel each step)")
Print(f"  dt={dt}")
Print(f"  collision ratio is {nu*dt}")
Print(f"  bins={bins}")
Print(f"  nsteps={nsteps}")
Print(f"  seed={seed}")
Print(f"  monitor_every={monitor_every}")
Print(f"  extra_collision={extra_collision}")
Print(f"  collision_type={collision_type}")
Print(f"  cross_section=hard_needle")
Print(f"  grazing_collision={grazing_collision}")

Print("--------------------------------------------------------------------")

info = {"inertia": 1.0,
        "mass": 1.0,
        "length": 1.0,
        "ev": 1.0,             # translational restitution
        "om": 1.0,             # rotational restitution
        "cutoff": 0.1,         # angular cutoff for near-parallel fallback
        "cross_section": "hard_needle",  # Onsager NTC kernel (arXiv:2508.10744 Example B)
       }

comm = MPI.COMM_WORLD
def vlasov_force(theta):
    local_nu_x = np.sum(np.cos(theta))
    local_nu_y = np.sum(np.sin(theta))
    global_nu_x = comm.allreduce(local_nu_x, op=MPI.SUM)
    global_nu_y = comm.allreduce(local_nu_y, op=MPI.SUM)
    angle_av = (np.arctan2(global_nu_y, global_nu_x) + 2*np.pi) % (2*np.pi)
    return -np.sin(theta-angle_av)

opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": extra_collision,
    "grazing_collision": grazing_collision,
    "collision_type": collision_type,
    "seed": seed,
    "test": "uniform_angle",
    "prefix": "output/test_17",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    vlasov_force=vlasov_force,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print(f"Simulation complete.  Final NTC running maximum: nu_max={sim._nu_max:.4e}")
