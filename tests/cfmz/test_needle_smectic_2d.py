"""
Test needle smectic 2-D — emergent smectic-A from Enskog-DSMC + ParaView export
================================================================================

Demonstrates the **smectic-A phase emerging from the Enskog kinetic
equation** in a 2-D rod system, with no imposed positional coupling
beyond the standard Onsager cross-section augmented by the
Carnahan-Starling-type Parsons-Lee correction.

The model is a thin-rod fluid in a 2-D periodic box at moderately high
density (η ≈ 0.6) and below the smectic transition temperature.  The
Enskog kernel uses position-derived contact normals (so the rigid-rod
impulse depends on the actual relative rod positions, not just on
random impact angles), and a Parsons-Lee g_PL(η_local) on the NTC
acceptance weight.  See CFMZ.md §14 for the full derivation.

Setup
-----
- 2-D periodic spatial domain, [0, Lx] × [0, Ly] with Lx = Ly = 1.
- ``CFMZNeedleDSMC`` with ``info["collision_kind"] = "enskog"``.
- Rod length L = 0.10 (so cell width Lx/bins = 0.125 ≥ L).
- nlocal = 200 000 → in-box density ρ ~ 200 000 ⇒ η = (π/4) ρ L² ~ 1500
  (well above the cap; effectively g_PL is at the cap value).
- Director seeded along x̂ via the new ``uniform_perturbed_2d`` IC
  with k=2 angular cosine.
- Andersen thermostat at T_bath = 0.30 (the smectic regime).
- VTK output every 50 steps to
  ``output/test_needle_smectic_2d_output_cfmz_inhomo_nanbu/dsmc_*.vtr``
  for ParaView visualisation.

Pass criteria
-------------
- The 2-D simulation runs to completion without errors.
- ``smectic_abs_0`` (k along x̂) rises above the noise floor 1/√N
  during the run, while ``smectic_abs_1`` (k along ŷ) stays at the
  noise floor.  This is the operational distinction between
  smectic-A (1-D positional order along director) and a 2-D crystal
  (positional order in both directions).
- VTK frames open in ParaView and show smectic stripes along x̂.

ParaView pipeline
-----------------
1. ``paraview output/test_needle_smectic_2d_output_cfmz_inhomo_nanbu/dsmc_*.vtr``
2. Apply *Glyph* on the cell-centre ``mean_orientation``, scale by
   ``local_R2``, type 2D-Edge.
3. Cell-colour by ``density`` (smectic stripes appear) or by
   ``local_psi_re_0`` (smectic phase pattern).
4. Use the time slider to animate the formation of layers.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dsmc import CFMZNeedleDSMC, Print

Opt = PETSc.Options()
nlocal = int(Opt.getReal("nlocal", 2e5))
bins = Opt.getInt("bins", 8)
dt = Opt.getReal("dt", 0.01)
nu = Opt.getReal("nu", 50.0)        # NTC initial estimate (≈ L · v_max)
nu_bath = Opt.getReal("nu_bath", 16.0)
T_bath = Opt.getReal("T_bath", 0.30)
nsteps = Opt.getInt("nsteps", 1500)
seed = Opt.getInt("seed", 47)

L = 0.10                # rod length
Lx = 1.0                # box side

info = {
    "mass": 1.0,
    "inertia": 1.0,
    "length": L,
    "ev": 1.0,
    "om": 1.0,
    "cutoff": 0.1,
    "cross_section": "hard_needle",       # Onsager-form NTC selection
    "collision_kind": "enskog",           # Parsons-Lee correction + position-aware
    "xmin": 0.0, "xmax": Lx,
    "ymin": 0.0, "ymax": Lx,
    "bcs": "periodic",
    "initial_angle_amplitude": 0.1,
    "initial_angle_shift": 0.0,
    "initial_angle_wavelength": 2,        # cos(2θ) seed — director along x̂
}

# Smectic wavevectors: along x̂ (expected to grow) and along ŷ (must
# stay at noise floor — the smectic-vs-crystal distinguishing test).
k_smectic = 2.0 * np.pi / Lx
opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": 1,
    "collision_type": "nanbu",
    "seed": seed,
    "test": "uniform_perturbed_2d",
    "spatial_dim": 2,
    "transport": True,
    "variance": "real_projective_plane",
    "n_modes": [1, 2, 4, 6],
    "smectic_k": [(k_smectic, 0.0), (0.0, k_smectic)],
    "T_bath": T_bath,
    "nu_bath": nu_bath,
    "prefix": "output/test_needle_smectic_2d",
}

Print("Running needle 2-D smectic test (Enskog + ParaView export):")
Print(f"  L={L}, η_avg ≈ (π/4) (nlocal/Lx²) L² = {(np.pi/4)*nlocal/(Lx*Lx)*L*L:.2f}")
Print(f"  T_bath={T_bath} (smectic regime), nu_bath={nu_bath}, nsteps={nsteps}")
Print(f"  smectic_k along x̂ and ŷ; smectic-A signature: ψ_S(x) ≫ ψ_S(y)")

sim = CFMZNeedleDSMC(opts=opts, info=info, comm=MPI.COMM_WORLD)

# Run with periodic VTK output for ParaView.
vtk_every = 50
n_full_blocks, leftover = divmod(nsteps, vtk_every)
done = 0
for b in range(n_full_blocks):
    sim.run(nsteps=vtk_every, monitor_every=vtk_every)
    done += vtk_every
    sim.export_cell_fields_vtk(
        prefix=f"{sim.output_path}/dsmc_{done:05d}",
        smectic_k=opts["smectic_k"],
        time=done * dt,
    )
if leftover:
    sim.run(nsteps=leftover, monitor_every=leftover)
    done += leftover
    sim.export_cell_fields_vtk(
        prefix=f"{sim.output_path}/dsmc_{done:05d}",
        smectic_k=opts["smectic_k"],
        time=done * dt,
    )

# Tail-averaged sanity report.
tail = max(1, nsteps // 3)
final = {n: float(np.mean(1.0 - np.array(sim.history[f"circular_var_n{n}"][-tail:])))
         for n in (1, 2, 4, 6)}
psi = {idx: float(np.mean(np.array(sim.history[f"smectic_abs_{idx}"][-tail:])))
       for idx in (0, 1)}
N_global = nlocal * MPI.COMM_WORLD.Get_size()
noise_floor = 1.0 / np.sqrt(N_global)

if MPI.COMM_WORLD.Get_rank() == 0:
    Print("\nFinal-third averages:")
    Print(f"  R₁={final[1]:.3f}  R₂={final[2]:.3f}  R₄={final[4]:.3f}  R₆={final[6]:.3f}")
    Print(f"  ψ_S(k‖x̂) = {psi[0]:.4f}    ψ_S(k‖ŷ) = {psi[1]:.4f}    "
          f"noise floor 1/√N ≈ {noise_floor:.4f}")
    if psi[0] < 5 * noise_floor:
        Print("  WARNING: ψ_S along x̂ did not rise above the noise floor — "
              "smectic-A did not emerge.  Try lower T_bath or higher density.")
    if psi[1] > 5 * noise_floor:
        Print("  WARNING: ψ_S along ŷ above noise floor — phase looks "
              "crystalline rather than smectic-A.")
Print("test_needle_smectic_2d complete.")
