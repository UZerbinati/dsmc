"""
2D flow past a cylinder — space-inhomogeneous Boltzmann with Maxwell molecules
------------------------------------------------------------------------------
Particles are initialised with a uniform Maxwellian + drift velocity u_inf in x.
Boundary conditions:
  - Periodic in x
  - Specular (elastic) reflection on the top/bottom walls
  - Specular (elastic) reflection on the cylinder surface

The simulation captures the formation of the density wake and temperature
variation around the cylinder as the gas evolves.

Run with:
    mpirun -n <P> python tests/boltzmann/test_3.py \
        -nlocal 200000 -bins 64 -nu 10 -dt 0.01 -nsteps 500 \
        -monitor_every 50 -inflow_velocity 1.5 -cylinder_radius 1.0
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import BoltzmannDSMC, Print

Opt = PETSc.Options()
Print("Running 2D Boltzmann DSMC — flow past a cylinder:")

nlocal            = int(Opt.getReal("nlocal", 2e5))
bins              = Opt.getInt("bins", 64)
dt                = Opt.getReal("dt", 0.01)
nu                = Opt.getReal("nu", 10.0)
nsteps            = Opt.getInt("nsteps", 500)
seed              = Opt.getInt("seed", 42)
collision_type    = Opt.getString("collision_type", "nanbu")
extra_collision   = Opt.getInt("extra_collision", 0) + 1
monitor_every     = Opt.getInt("monitor_every", 50)
inflow_velocity   = Opt.getReal("inflow_velocity", 1.5)
cylinder_radius   = Opt.getReal("cylinder_radius", 1.0)

Print(f"  nlocal={nlocal}")
Print(f"  bins={bins}")
Print(f"  nu={nu}")
Print(f"  dt={dt}")
Print(f"  collision ratio is {nu*dt}")
Print(f"  nsteps={nsteps}")
Print(f"  seed={seed}")
Print(f"  collision_type={collision_type}")
Print(f"  extra_collision={extra_collision}")
Print(f"  monitor_every={monitor_every}")
Print(f"  inflow_velocity={inflow_velocity}")
Print(f"  cylinder_radius={cylinder_radius}")
Print("--------------------------------------------------------------------")

info = {
    "temperature":        1.0,
    "mass":               1.0,
    "inflow_velocity":    inflow_velocity,
    "cylinder_radius":    cylinder_radius,
    "cylinder_center_x":  0.0,
    "cylinder_center_y":  0.0,
    "xmin": -8.0,
    "xmax": 12.0,
    "ymin": -5.0,
    "ymax":  5.0,
}
opts = {
    "nlocal":          nlocal,
    "nu":              nu,
    "dt":              dt,
    "bins":            bins,
    "extra_collision": extra_collision,
    "collision_type":  collision_type,
    "seed":            seed,
    "test":            "cylinder_flow",
    "prefix":          "output/test_3",
}
sim = BoltzmannDSMC(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")
