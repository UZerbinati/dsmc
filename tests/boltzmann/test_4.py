"""
2D flow past a cylinder — space-inhomogeneous Boltzmann with Maxwell molecules
------------------------------------------------------------------------------
Particles are initialised with a uniform Maxwellian + drift velocity u_inf in x.
Boundary conditions:
  - Periodic in x
  - Specular (elastic) reflection on the top/bottom walls
  - Specular (elastic) reflection on the cylinder surface

Expected flow regime (default parameters: nu=10, u_inf=1.5, R=1):
  - Mean free path:   lambda ~ v_th / nu = 1/10 = 0.1
  - Knudsen number:   Kn = lambda / R ~ 0.1  (slip/transitional regime)
  - Kinematic visc.:  mu ~ T / (m * nu) = 0.1
  - Reynolds number:  Re = u_inf * R / mu ~ 15  (laminar)

  At Re ~ 15 the flow is well within the laminar regime (vortex shedding
  begins only around Re ~ 40-50).  You should observe:
    - A symmetric laminar wake behind the cylinder
    - A density shadow downstream and slight compression upstream
    - Temperature variation: heating at the stagnation point, cooling at
      the sides

  To reach the vortex-shedding or turbulent regime you would need to raise
  nu substantially (lowering Kn and raising Re), use a larger domain, and
  increase the particle count to keep statistical noise below the physical
  fluctuations.  Note also that this is a 2D simulation: classical 3D
  turbulence cannot develop; only 2D turbulence (inverse energy cascade)
  would be possible at sufficiently high Re.

Run with:
    mpirun -n <P> python tests/boltzmann/test_4.py \
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

nlocal            = int(Opt.getReal("nlocal", 1e7))
bins              = Opt.getInt("bins", 512)
dt                = Opt.getReal("dt", 0.01)
nu                = Opt.getReal("nu", 10.0)
nsteps            = Opt.getInt("nsteps", 1000)
seed              = Opt.getInt("seed", 42)
collision_type    = Opt.getString("collision_type", "nanbu")
extra_collision   = Opt.getInt("extra_collision", 0) + 1
monitor_every     = Opt.getInt("monitor_every", 10)
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
    "prefix":          "output/test_4",
}
sim = BoltzmannDSMC(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")
