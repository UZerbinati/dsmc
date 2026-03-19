"""
Test with with Vlasov force
----------------------------
Here we consdier a quadratic potential reuslting in the Valsov force
V = -alpha(theta-theta_av)
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMDSMC, Print
import numpy as np

Opt = PETSc.Options()
Print("Running homogeneous CFM DSMC with options:")

nlocal = Opt.getReal("nlocal", 1e7)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.01)
nu = Opt.getReal("nu", 100)
nsteps = Opt.getInt("nsteps", 2000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
monitor_every = Opt.getInt("monitor_every", 10)

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
Print(f"  grazing_collision={grazing_collision}")

Print("--------------------------------------------------------------------")

#TODO: Fix with correct relation between length mass and inertia
info = {"inertia": 1.0,
        "mass": 1.0,
        "length": 1.0,
        "ev": 1.0,       # translational restitution
        "om": 1.0,       # rotational restitution
        "cutoff": 0.1,   # angular cutoff
       }
vlasov_force = lambda theta, theta_av: -np.sin(theta-theta_av)
sim = CFMDSMC(
    nlocal=nlocal,
    nu=nu,
    dt=dt,
    bins = bins,
    info=info,
    extra_collision=extra_collision,
    grazing_collision=grazing_collision,
    collision_type=collision_type,
    vlasov_force = vlasov_force,
    seed=seed,
    test="uniform_angle",
    prefix="test_7",
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")

