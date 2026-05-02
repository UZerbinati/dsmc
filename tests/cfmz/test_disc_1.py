"""
Discotic test 1 — NVE discotic-nematic emergence (3-D-disc-in-2-D)
==================================================================

Models a 3-D discotic LC simulated in 2-D with the calamitic-form
Onsager kernel ``W = |sin(Δθ)|`` (``opts["n_fold"] = 2`` default), where
θ is the projected disc normal.  Hard-disc collisions, no thermostat
(microcanonical / NVE).  Initial orientation density is perturbed with a
**2-fold** cosine seed,

    ρ(θ, t=0) ∝ 1 + A cos(2 θ + φ),    A = 0.1.

so the unstable cos(2θ) mode of the I-N spinodal grows from a finite
seed.  This recovers the classical Onsager I-N transition with the
nematic order parameter R₂ as the critical mode — but here R₂
characterises alignment of *disc normals* (= discotic nematic N_D),
not of rod long axes.

Pass criteria
-------------
- R₂(t) grows from ≈ A/2 ≈ 0.05 to a steady plateau in ≈ 0.6–0.9.
- R₁, R₄, R₆ stay near zero throughout (only the head-tail-symmetric
  R₂ is driven by the |sin(Δθ)| kernel).
- Total energy E_kin/N + ½ L² E[ρ] is conserved within 1 %.
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
nsteps = Opt.getInt("nsteps", 1500)
seed = Opt.getInt("seed", 47)

R = np.sqrt(12.0)   # makes T_c = R²·2/(3π) = 8/π ≈ 2.55 (spinodal) — see CFMZ.md §13.2
info = {
    "mass": 1.0,
    "inertia": 1.0,
    "radius": R,
    "ev": 1.0,
    "om": 1.0,
    "cross_section": "hard_disc",
    "initial_angle_amplitude": 1e-1,
    "initial_angle_shift": -0.3,
    "initial_angle_wavelength": 2,    # cos(2θ) seed for the nematic mode
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
    "prefix": "output/test_disc_1",
}

Print("Running discotic test 1 — NVE discotic-nematic emergence (N_D):")
Print(f"  nlocal={nlocal}, nu={nu}, dt={dt}, nsteps={nsteps}, seed={seed}")
Print(f"  R={R:.3f},  α_c·R⁻²·T_c = 8/π ≈ {8/np.pi:.3f}  (spinodal estimate)")

sim = CFMZDiscDSMCHomo(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=200)
Print("test_disc_1 complete.")
