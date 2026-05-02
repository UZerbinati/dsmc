"""
Discotic test 5 — head-tail symmetry null check (sanity)
==========================================================

Same setup as ``test_disc_3`` (NVT, T_bath = 0.5, deeply nematic in
the 3-D-disc-in-2-D model).  The check is the head-tail-symmetry
signal: **R₁ must stay near zero** even when R₂ saturates near 1.

Note on the higher harmonics R₄, R₆: these are NOT independent
diagnostics in a deeply ordered nematic phase.  For an aligned
distribution ρ(θ) ∝ exp[βV cos(2θ)] the moments follow
⟨cos(2nθ)⟩ → 1 monotonically with βV in the ordered limit, so all
even harmonics are close to 1.  Only R₁ ≈ 0 is the *symmetry*
signature of head-tail invariance (θ ≡ θ+π); the other Rₙ values
just trace the angular distribution width.

Pass criteria
-------------
- R₁ stays below 0.05 throughout (clean head-tail symmetry).
- R₂ exceeds 0.85 in the final third (deep nematic order).
- R₂ > R₄ > R₆ (Bessel-ratio ordering for a thermal nematic).
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
    "initial_angle_wavelength": 2,
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

Print("Running discotic test 5 — head-tail symmetry null check:")
Print(f"  T_bath={T_bath} (deeply nematic); R₁ must remain near zero")

sim = CFMZDiscDSMCHomo(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=200)

# Post-run sanity print: tail-averaged R_n.
tail = max(1, nsteps // 3)
final = {n: float(np.mean(1.0 - np.array(sim.history[f"circular_var_n{n}"][-tail:])))
         for n in (1, 2, 4, 6)}
Print(f"\nFinal-third averages:  R₁={final[1]:.3f}  R₂={final[2]:.3f}  "
      f"R₄={final[4]:.3f}  R₆={final[6]:.3f}")
if MPI.COMM_WORLD.Get_rank() == 0:
    if final[1] > 0.05:
        Print("  WARNING: R₁ above head-tail-symmetry noise threshold.")
    if final[2] < 0.85:
        Print("  WARNING: R₂ did not saturate into the deeply nematic regime.")
    if not (final[2] > final[4] > final[6]):
        Print("  WARNING: harmonics not in expected Bessel-ratio ordering "
              "(R₂ > R₄ > R₆ for a thermal nematic).")
Print("test_disc_5 complete.")
