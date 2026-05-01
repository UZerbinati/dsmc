"""
Discotic test 6 — inhomogeneous solver, smectic diagnostic
===========================================================

End-to-end integration test for ``CFMZDiscDSMC`` (inhomogeneous discotic
solver).  Runs a 1-D periodic spatial domain at low T_bath (deeply
tetratic regime) with a uniform initial density, and tracks both the
orientational order family ``R_n`` and the smectic / positional order
parameter ``ψ_S(k) = |⟨e^{i k x}⟩|`` at the fundamental and first
harmonic wavevectors of the spatial box.

What the test verifies
----------------------
1.  ``CFMZDiscDSMC`` instantiates and runs without errors using the
    inhomogeneous infrastructure (per-cell Nanbu, position migration,
    cell-DM coordinate access).
2.  The discotic Vlasov force still drives the I-T transition with
    positions present: ``R₄`` reaches the expected tetratic plateau.
3.  The new smectic diagnostic is computed correctly.  With a uniform
    spatial IC and no position-orientation coupling in the dynamics,
    the columnar order parameters ``ψ_S(k)`` should stay at their
    finite-N noise floor ``≈ 1/√N`` throughout the run (since the
    density remains spatially uniform).  This is the cleanest possible
    unit test for the diagnostic.

For a true positional/columnar phase, additional position-orientation
coupling — e.g. a ``translational_force`` callable, a non-uniform IC,
or boundary forcing — would be needed.  That is left as research
follow-up; the infrastructure built here makes it straightforward.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dsmc import CFMZDiscDSMC, Print

Opt = PETSc.Options()
nlocal = int(Opt.getReal("nlocal", 1e5))
bins = Opt.getInt("bins", 64)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4.0)
# nu_bath = 16 (vs 4 in the homogeneous tests) keeps the Andersen
# thermostat strong enough to balance the per-cell Vlasov heating
# under inhomogeneous transport.
nu_bath = Opt.getReal("nu_bath", 16.0)
T_bath = Opt.getReal("T_bath", 0.5)
nsteps = Opt.getInt("nsteps", 800)
seed = Opt.getInt("seed", 47)

R = np.sqrt(12.0)
Lx = 1.0

info = {
    "mass": 1.0,
    "inertia": 1.0,
    "radius": R,
    "ev": 1.0,
    "om": 1.0,
    "cross_section": "hard_disc",
    "Lx": Lx,
    "bcs": "periodic",
}

opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": 1,
    "collision_type": "nanbu",
    "seed": seed,
    "test": "uniform_1d",
    "spatial_dim": 1,
    "transport": True,
    "variance": "real_projective_plane",
    "n_modes": [1, 2, 4, 6],
    # Two wavevectors: fundamental and 2nd harmonic of the spatial box.
    "smectic_k": [(2 * np.pi / Lx,), (4 * np.pi / Lx,)],
    "T_bath": T_bath,
    "nu_bath": nu_bath,
    "prefix": "output/test_disc_6",
}

Print("Running discotic test 6 — inhomogeneous solver + smectic diagnostic:")
Print(f"  nlocal={nlocal}, T_bath={T_bath} (deeply tetratic, α≈{R*R/T_bath:.2f})")
Print(f"  smectic_k = [(2π/Lx,), (4π/Lx,)]  with  Lx={Lx}")

sim = CFMZDiscDSMC(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=200)

# Tail-averaged sanity check over the final third of the run.
tail = max(1, nsteps // 3)
final = {n: float(np.mean(1.0 - np.array(sim.history[f"circular_var_n{n}"][-tail:])))
         for n in (1, 2, 4, 6)}
psi = {idx: float(np.mean(np.array(sim.history[f"smectic_abs_{idx}"][-tail:])))
       for idx in (0, 1)}
N_global = nlocal * MPI.COMM_WORLD.Get_size()
noise_floor = 1.0 / np.sqrt(N_global)

if MPI.COMM_WORLD.Get_rank() == 0:
    Print(f"\nFinal-third averages:")
    Print(f"  R₁={final[1]:.3f}  R₂={final[2]:.3f}  R₄={final[4]:.3f}  R₆={final[6]:.3f}")
    Print(f"  ψ_S(k₀)={psi[0]:.4f}  ψ_S(k₁)={psi[1]:.4f}  noise floor 1/√N≈{noise_floor:.4f}")
    if final[4] < 0.5:
        Print("  WARNING: tetratic order R₄ did not develop as expected.")
    if psi[0] > 5 * noise_floor:
        Print("  WARNING: smectic order at k₀ above expected noise floor.")
Print("test_disc_6 complete.")
