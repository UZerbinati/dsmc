"""
Discotic test 0 — pure-collision relaxation (3-D-disc-in-2-D)
=============================================================

Models a 3-D discotic LC simulated in a 2-D domain: each particle's
orientation θ is the in-plane projection of the disc normal, so
θ ≡ θ + π (head-tail symmetric).  The Onsager kernel ``|sin(Δθ)|``
(``opts["n_fold"] = 2``, the new default) drives the discotic-nematic
phase N_D when the mean field is active.

This test runs hard 2-D disc collisions with **no** Vlasov force and
**no** Andersen thermostat — so the orientations stay uniform and the
test focuses on the *collision invariants*.  Standard parameters:
nlocal=2.5e5, bins=256, dt=0.05, nsteps=500.

Pass criteria
-------------
- Total kinetic energy conserved within 1e-3 (elastic collisions, ev=1).
- Hard-disc collisions do **not** transfer torque, so the angular-velocity
  distribution shape is unchanged.  Numerically: ⟨ω²⟩ stays constant
  within 1 % of its t=0 value, and the rotational energy is a
  separately-conserved invariant.
- All R_n stay below ≈ 3 σ statistical noise from the uniform-angle
  baseline (R_n_uniform = 0).

The "no torque transfer" property follows from the contact-arm geometry
``r_i = -R n``, ``r_j = +R n`` ⇒ ``c_k = r_k × n = 0``; see CFMZ.md §13.3.
The collision rule does NOT depend on the orientational kernel choice
(n_fold), only on disc geometry.
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
nsteps = Opt.getInt("nsteps", 500)
seed = Opt.getInt("seed", 47)

Print("Running discotic test 0 — pure-collision relaxation:")
Print(f"  nlocal={nlocal}, nu={nu}, dt={dt}, nsteps={nsteps}, seed={seed}")
Print(f"  vlasov_force=False (disabled), cross_section=hard_disc")

R = 0.5
info = {
    "mass": 1.0,
    "inertia": 1.0,                 # natural-units convention (matches calamitic tests)
    "radius": R,
    "ev": 1.0,
    "om": 1.0,
    "cross_section": "hard_disc",
}
opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": 1,
    "collision_type": "nanbu",
    "seed": seed,
    "test": "uniform_angle",
    "variance": "real_projective_plane",
    "n_modes": [1, 2, 4, 6],
    "prefix": "output/test_disc_0",
}

# vlasov_force=False  → explicitly disabled (no mean-field torque).
# interaction_energy=False → not tracked (no E[ρ] history).
sim = CFMZDiscDSMCHomo(
    opts=opts,
    info=info,
    vlasov_force=False,
    interaction_energy=False,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=nsteps)
Print("test_disc_0 complete.")
