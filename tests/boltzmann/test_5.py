"""
2D flow past a cylinder — vortex shedding regime (von Kármán vortex street)
----------------------------------------------------------------------------
Same geometry as test_4 but with a higher collision frequency to push the
flow into the vortex-shedding regime.

Dimensionless parameters (default values):
  - Mean free path:   lambda ~ v_th / nu = 1/50 = 0.02
  - Knudsen number:   Kn = lambda / R ~ 0.02  (near-continuum)
  - Kinematic visc.:  mu ~ T / (m * nu) = 1/50 = 0.02
  - Reynolds number:  Re = u_inf * D / mu = 1.5 * 2 / 0.02 = 150

  At Re ~ 150 the flow is in the vortex-shedding regime (onset is around
  Re ~ 40-50).  You should observe a von Kármán vortex street: alternating
  vortices shed from the top and bottom of the cylinder and advected
  downstream.  The shedding frequency is characterised by the Strouhal
  number St ~ 0.2:
      f_shed ~ St * u_inf / D ~ 0.2 * 1.5 / 2 = 0.15
      T_shed ~ 1 / f_shed ~ 6.7 time units
  Running for nsteps=4000 with dt=0.005 covers t=20 ~ 3 shedding periods,
  which is enough to establish the periodic vortex street.

Statistical noise:
  With nlocal=2e6 per rank and bins=128, there are roughly
      N_cell ~ (nlocal * nranks) / bins^2 ~ 600 particles per cell.
  The DSMC noise is ~1/sqrt(N_cell) ~ 4%, well below the vortex velocity
  fluctuations (~30-50% of u_inf).

Why not turbulence?
  True turbulence (Re > ~500) would require nu >= 250, dt <= 0.004, and
  N_cell >= 10,000 to resolve the much smaller turbulent fluctuations above
  the statistical noise floor.  This is computationally impractical on a
  workstation; a large HPC run would be needed.

Run with:
    mpirun -n <P> python tests/boltzmann/test_5.py \
        -nlocal 2000000 -bins 128 -nu 50 -dt 0.005 -nsteps 4000 \
        -monitor_every 100 -inflow_velocity 1.5 -cylinder_radius 1.0
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import BoltzmannDSMC, Print

Opt = PETSc.Options()
Print("Running 2D Boltzmann DSMC — flow past a cylinder (vortex shedding):")

nlocal          = int(Opt.getReal("nlocal", 1e7))
bins            = Opt.getInt("bins", 512)
dt              = Opt.getReal("dt", 0.005)
nu              = Opt.getReal("nu", 50.0)
nsteps          = Opt.getInt("nsteps", 4000)
seed            = Opt.getInt("seed", 42)
collision_type  = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0) + 1
monitor_every   = Opt.getInt("monitor_every", 100)
inflow_velocity = Opt.getReal("inflow_velocity", 1.5)
cylinder_radius = Opt.getReal("cylinder_radius", 1.0)

comm = MPI.COMM_WORLD
nranks = comm.Get_size()
Re = inflow_velocity * 2 * cylinder_radius * nu  # Re = u*D/mu, mu=1/nu

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
Print(f"  estimated Re ~ {Re:.1f}")
Print(f"  estimated N_cell ~ {nlocal * nranks / bins**2:.0f}")
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
    "prefix":          "output/test_5",
}
sim = BoltzmannDSMC(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")
