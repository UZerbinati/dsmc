"""
Test with no Vlasov force
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMDSMC, Print

Opt = PETSc.Options()
Print("Running homogeneous CFM DSMC with options:")

nlocal = Opt.getReal("nlocal", 20000)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 31)
dt = Opt.getReal("dt", 1e-2)
nu = Opt.getReal("nu", 1/dt)
nsteps = Opt.getInt("nsteps", 200)
seed = Opt.getInt("seed", 1234)
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
sim = CFMDSMC(
    nlocal=nlocal,
    nu=nu,
    dt=dt,
    bins = bins,
    info=info,
    extra_collision=extra_collision,
    grazing_collision=grazing_collision,
    collision_type=collision_type,
    seed=seed,
    prefix="test_1",
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")

