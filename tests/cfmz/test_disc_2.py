"""
Discotic test 2 — isotropic regime under Andersen thermostat
=============================================================

3-D-disc-in-2-D model (n_fold=2 default, kernel ``|sin(Δθ)|``) with
Andersen thermostat at T_bath = 8.0.  At this temperature the
dimensionless Onsager coupling α = R²/T_bath ≈ 1.5 is well below the
spinodal α_c = 3π/2 ≈ 4.71, so orientations should remain isotropic.

Pass criteria
-------------
- All R_n stay below 0.10 in steady state (after t > 5 / nu_bath).
- Kinetic temperature T → T_bath within 5 %.
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
T_bath = Opt.getReal("T_bath", 8.0)
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
    "initial_angle_wavelength": 2,    # cos(2θ) seed — same as test_disc_1
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
    "prefix": "output/test_disc_2",
}

Print("Running discotic test 2 — NVT isotropic regime:")
Print(f"  T_bath={T_bath} (α = R²/T_bath = {R*R/T_bath:.2f}), α_c≈{3*np.pi/2:.2f}")
Print(f"  nlocal={nlocal}, nu={nu}, nu_bath={nu_bath}, dt={dt}, nsteps={nsteps}")

sim = CFMZDiscDSMCHomo(opts=opts, info=info, comm=MPI.COMM_WORLD)
sim.run(nsteps=nsteps, monitor_every=200)
Print("test_disc_2 complete.")
