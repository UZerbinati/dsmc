"""
Sod shock tube test — space-inhomogeneous Boltzmann with Maxwell molecules
---------------------------------------------------------------------------
Left state:  rho=1,     T=1.0  (x < 0.5)
Right state: rho=0.125, T=0.8  (x > 0.5)

Run with:
    mpirun -n <P> python tests/boltzmann/test_0.py -nlocal 10000 -nsteps 200 \
        -monitor_every 20 -bins 64 -nu 100 -dt 0.001
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import BoltzmannDSMC, Print

Opt = PETSc.Options()
Print("Running space-inhomogeneous Maxwell-molecule DSMC (Sod shock tube):")

nlocal = Opt.getReal("nlocal", 1e7)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.01)
nu = Opt.getReal("nu", 10)
nsteps = Opt.getInt("nsteps", 2000)
seed = Opt.getInt("seed", 42)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0) + 1
monitor_every = Opt.getInt("monitor_every", 5)

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
Print("--------------------------------------------------------------------")

opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": extra_collision,
    "collision_type": collision_type,
    "seed": seed,
    "test": "sod",
    "prefix": "output/test_0",
}
info = {
    "temperature": 1.0,
    "mass": 1.0,
}
sim = BoltzmannDSMC(
    opts=opts,
    info=info,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")
