"""
Quadratic Vlasov force, α=4
----------------------------
Mean-field torque: F(θ) = −4(θ − θ_av), where θ_av is the mean
orientation angle computed via circular statistics.  Same as test_2
with a stronger confinement strength α=4.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print
import numpy as np

Opt = PETSc.Options()
Print("Running homogeneous CFMZ needle DSMC with options:")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4)
nsteps = Opt.getInt("nsteps", 4000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
monitor_every = Opt.getInt("monitor_every", 100)

Print(f"  nlocal={nlocal}")
Print(f"  nu={nu}")
Print(f"  dt={dt}")
Print(f"  collision ratio is {nu*dt}")
Print(f"  bins={bins}")
Print(f"  nsteps={nsteps}")
Print(f"  seed={seed}")
Print(f"  monitor_every={monitor_every}")
Print(f"  extra_collision={extra_collision}")
Print(f"  collision_type={collision_type}")
Print(f"  cross_section=maxwell")
Print(f"  grazing_collision={grazing_collision}")

Print("--------------------------------------------------------------------")

#TODO: Fix with correct relation between length mass and inertia
info = {"inertia": 1.0,
        "mass": 1.0,
        "length": 1.0,
        "ev": 1.0,       # translational restitution
        "om": 1.0,       # rotational restitution
        "cutoff": 0.1,   # angular cutoff
        "cross_section": "maxwell",
       }
comm = MPI.COMM_WORLD
def vlasov_force(theta):
    local_nu_x = np.sum(np.cos(theta))
    local_nu_y = np.sum(np.sin(theta))
    global_nu_x = comm.allreduce(local_nu_x, op=MPI.SUM)
    global_nu_y = comm.allreduce(local_nu_y, op=MPI.SUM)
    angle_av = (np.arctan2(global_nu_y, global_nu_x) + 2*np.pi) % (2*np.pi)
    return -4*(theta-angle_av)

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
    "prefix": "output/test_3",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    vlasov_force=vlasov_force,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")

