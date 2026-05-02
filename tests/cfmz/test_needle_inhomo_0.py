"""
Sod-like rod shock tube — non-space-homogeneous CFMZ
-----------------------------------------------------
Left state  (x < Lx/2): rho=1,     T=1.0, theta uniform on (0, 2π)
Right state (x > Lx/2): rho=0.125, T=0.8, theta vonMises(π/2, kappa)

Run with:
    mpirun -n <P> python tests/cfmz/test_inhomo_0.py -nlocal 1e6 -nsteps 200 \
        -monitor_every 20 -bins 128 -nu 10 -dt 0.01
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print

Opt = PETSc.Options()
Print("Running non-space-homogeneous CFMZ needle DSMC (Sod-like rod shock):")

nlocal = int(Opt.getReal("nlocal", 1e6))
bins = Opt.getInt("bins", 128)
dt = Opt.getReal("dt", 0.01)
nu = Opt.getReal("nu", 10)
nsteps = Opt.getInt("nsteps", 200)
seed = Opt.getInt("seed", 47)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0) + 1
monitor_every = Opt.getInt("monitor_every", 20)

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

info = {
    "inertia": 1.0,
    "mass": 1.0,
    "length": 0.1,
    "ev": 1.0,
    "om": 1.0,
    "cutoff": 0.1,
    "cross_section": "maxwell",
    "Lx": 1.0,
    "bcs": "reflective",
    "right_concentration": 2.0,
}
opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": extra_collision,
    "collision_type": collision_type,
    "seed": seed,
    "test": "sod_rod",
    "spatial_dim": 1,
    "transport": True,
    "prefix": "output/test_needle_inhomo_0",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")
