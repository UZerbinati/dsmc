"""
Hard-needle cross-section — NTC Nanbu collision step
-----------------------------------------------------
The collision kernel for 2-D calamitic (needle-shaped) molecules derived in
Example B of arXiv:2508.10744 is

    W(Ξ₁, Ξ₂ → Ξ'₁, Ξ'₂) = |g·n| · S(ν₁, ν₂)

where g is the effective contact velocity (eq. 4.5 of the paper) and

    S(ν₁, ν₂) = L |sin(θ₁ − θ₂)|

is the Onsager excluded-volume cross-section for 2-D needles.  Near-parallel
rods (|sin(Δθ)| ≈ 0) have a vanishingly small cross-section and are almost
never selected for collision, in contrast to the Maxwell (uniform kernel) case
used in tests 0–12.

Bird's No-Time-Counter (NTC) acceptance–rejection method is applied:
  - Mcol_cand = floor(ν_max · N · dt / 2) candidate pairs are drawn each step.
  - Each candidate is accepted with probability |g·n| · L|sin(Δθ)| / ν_max.
  - The running maximum ν_max (stored as sim._nu_max) is updated every step.

Compare with test_1 (same parameters, Maxwell kernel) to observe the effect of
the anisotropic cross-section on equilibration.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print

Opt = PETSc.Options()
Print("Running homogeneous CFMZ needle DSMC — hard-needle cross-section (NTC):")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 10)
nsteps = Opt.getInt("nsteps", 1000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
monitor_every = Opt.getInt("monitor_every", 100)

Print(f"  nlocal={nlocal}")
Print(f"  nu={nu}  (initial NTC estimate; adapts to max kernel each step)")
Print(f"  dt={dt}")
Print(f"  bins={bins}")
Print(f"  nsteps={nsteps}")
Print(f"  seed={seed}")
Print(f"  monitor_every={monitor_every}")
Print(f"  extra_collision={extra_collision}")
Print(f"  collision_type={collision_type}")
Print(f"  grazing_collision={grazing_collision}")
Print(f"  cross_section=hard_needle  (kernel: L|sin(θ₁-θ₂)| · |g·n|)")

Print("--------------------------------------------------------------------")

info = {"inertia": 1.0,
        "mass": 1.0,
        "length": 1.0,
        "ev": 1.0,             # translational restitution
        "om": 1.0,             # rotational restitution
        "cutoff": 0.1,         # angular cutoff for near-parallel fallback
        "cross_section": "hard_needle",  # Onsager NTC kernel (arXiv:2508.10744 Example B)
       }
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
    "prefix": "output/test_13",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print(f"Simulation complete.  Final NTC running maximum: nu_max={sim._nu_max:.4e}")
