"""
Discotic test 5 — polar / hexatic null check (sanity)
======================================================

Same setup as ``test_disc_3`` (NVT, T_bath = 0.5, deeply tetratic),
but the post-run check is on the *suppressed* harmonics R₁ and R₆.

Pass criteria
-------------
- R₁ stays below 0.10 throughout (head-tail symmetry preserved by the
  W = |sin(2 Δθ)| kernel).
- R₆ stays below 0.15 throughout (no spurious 6-fold ordering).
- R₄ ≫ R₁, R₆ over the final third of the run.

This is the cleanest check that the 4-fold-symmetric kernel does not
inadvertently couple to other rotational symmetries.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dsmc import CFMZDiscDSMCHomo, Print

Opt = PETSc.Options()
nlocal = int(Opt.getReal("nlocal", 2.5e5))
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4.0)
nu_bath = Opt.getReal("nu_bath", 4.0)
T_bath = Opt.getReal("T_bath", 0.5)
nsteps = Opt.getInt("nsteps", 1500)
seed = Opt.getInt("seed", 47)

R = np.sqrt(12.0)
info = {
    "mass": 1.0,
    "inertia": 1.0,
    "radius": R,
    "ev": 1.0,
    "om": 1.0,
    "cross_section": "hard_disc",
    "initial_angle_amplitude": 1e-1,
    "initial_angle_shift": -0.3,
    "initial_angle_wavelength": 4,
}
opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": 1,
    "collision_type": "nanbu",
    "seed": seed,
    "test": "perturbed_uniform_angle",
    "variance": "real_projective_plane",
    "n_modes": [1, 2, 4, 6],
    "T_bath": T_bath,
    "nu_bath": nu_bath,
    "prefix": "output/test_disc_5",
}

Print("Running discotic test 5 — polar / hexatic null check:")
Print(f"  T_bath={T_bath} (deeply tetratic), watching R₁ and R₆ for spurious order")

sim = CFMZDiscDSMCHomo(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=200)

# Post-run sanity print: tail-averaged R_n.
tail = max(1, nsteps // 3)
final = {n: float(np.mean(1.0 - np.array(sim.history[f"circular_var_n{n}"][-tail:])))
         for n in (1, 2, 4, 6)}
Print(f"\nFinal-third averages:  R₁={final[1]:.3f}  R₂={final[2]:.3f}  "
      f"R₄={final[4]:.3f}  R₆={final[6]:.3f}")
if MPI.COMM_WORLD.Get_rank() == 0:
    if final[1] > 0.10 or final[6] > 0.15:
        Print("  WARNING: spurious R₁ or R₆ order detected!")
    if final[4] < 5 * max(final[1], final[6]):
        Print("  WARNING: R₄ is not strongly dominant over R₁/R₆.")
Print("test_disc_5 complete.")
