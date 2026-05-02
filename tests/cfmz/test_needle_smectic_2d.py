"""
Test needle smectic 2-D — emergent smectic-A from Onsager + Enskog DSMC
========================================================================

Demonstrates the **smectic-A phase** in a 2-D rod system.  The
orientational drive is the **classical Onsager mean-field** with kernel
``W = |sin(Δθ)|`` (same as ``test_needle_12.py``); on top of that the
**Enskog kernel** restores the position-orientation coupling that
turns the nematic into a smectic.  See CFMZ.md §14 for the full
physics derivation.

Why both: pure Onsager mean-field gives the I-N transition but no
positional ordering; pure Enskog has the positional coupling but no
temperature-dependent driving force.  Together they produce the
I → N → Sm-A sequence.

Parameter philosophy
--------------------
- ``L = √12`` so that the Onsager spinodal sits at the canonical
  ``T_NI = 8/π ≈ 2.55`` (matches the rest of the calamitic suite).
- Box ``Lx = n_layers · L`` with ``n_layers`` rod-lengths across so
  the smectic wavelength fits commensurately.  Default ``n_layers=50``.
- ``nlocal`` scaled to give a target Parsons-Lee packing
  ``η = (π/4) ρ L² ≈ 0.5`` so the density correction is *active*
  (not capped); this is the regime where Enskog adds new physics
  beyond Boltzmann.
- ``bins`` chosen so cell width ≥ L (cell-sizing constraint, §14.5).

Setup
-----
- 2-D periodic box, `spatial_dim = 2`, `uniform_perturbed_2d` IC.
- Director seeded along x̂ via the ``cos(2θ)`` perturbation.
- Andersen thermostat at ``T_bath = 0.5`` (deep below T_NI, smectic
  regime if it exists).
- ``info["collision_kind"] = "enskog"`` — Parsons-Lee correction +
  position-derived contact normal.
- Vlasov force: ``calamitic Onsager closure`` (built inline; see
  ``test_needle_12.py`` for the homogeneous version).  Enable with
  ``-vlasov 1`` (default), disable with ``-vlasov 0`` to test the
  pure-Enskog limit.

Smectic wavevectors
-------------------
Five wavevectors around the natural smectic mode
``k_natural = 2π / L`` are registered along x̂ and another five along
ŷ.  In a true smectic-A phase, the projection onto x̂ has a sharp
peak at the layer-spacing wavevector while the ŷ projections stay at
the noise floor (the smectic-vs-crystal distinguishing diagnostic;
§14.8).

VTK output every 50 steps to
``output/test_needle_smectic_2d_output_cfmz_inhomo_nanbu/dsmc_*.vtr``.

Pass criteria
-------------
- Run completes without errors.
- ``smectic_abs_<idx>`` for *some* x-aligned wavevector rises above
  noise; *all* y-aligned wavevectors stay at noise.
- VTK frames in ParaView show density stripes along x̂.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dsmc import CFMZNeedleDSMC, Print

Opt = PETSc.Options()
n_layers = Opt.getInt("n_layers", 50)
eta_target = Opt.getReal("eta", 0.5)
bins = Opt.getInt("bins", 8)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 20.0)
nu_bath = Opt.getReal("nu_bath", 16.0)
T_bath = Opt.getReal("T_bath", 0.5)
nsteps = Opt.getInt("nsteps", 3000)
seed = Opt.getInt("seed", 47)
use_vlasov = bool(Opt.getInt("vlasov", 1))   # 1 = on (default), 0 = pure Enskog

# Physical-units choices.
L = np.sqrt(12.0)                       # rod length — sets T_NI = 8/π
Lx = float(n_layers) * L                # box: n_layers rod-lengths across
# nlocal_total chosen from η_target = (π/4) ρ L² with ρ = N/Lx²:
#   N = (4/π) η · Lx²/L² = (4/π) η · n_layers²
N_total = max(int(round((4.0 / np.pi) * eta_target * n_layers ** 2)), 100)
nlocal = max(N_total // max(1, MPI.COMM_WORLD.Get_size()), 1)


def _onsager_vlasov_inhomo(L_rod, bins_theta, comm):
    """Build (vlasov_force, interaction_energy) closures for the
    classical calamitic Onsager kernel ``W = |sin Δθ|`` adapted to
    the inhomogeneous-solver signature.

    Mirrors the closure in ``test_needle_12.py`` (homogeneous) but
    wraps the force so it can be passed to ``CFMZNeedleDSMC``: the
    inhomogeneous Vlasov-kick step calls
    ``vlasov_force(angle, X, density)`` — we ignore X and density
    here and use a *global* CIC-allreduced θ-density, identical to
    the homogeneous mean-field treatment.
    """
    delta = 2 * np.pi / bins_theta
    centers = (np.arange(bins_theta) + 0.5) * delta
    diff = centers[:, None] - centers[None, :]
    W_mat = np.abs(np.sin(diff))

    def cic_density_global(theta_local):
        t = theta_local.ravel() / delta
        k = np.floor(t).astype(int) % bins_theta
        w2 = t - np.floor(t)
        w1 = 1.0 - w2
        rho = np.zeros(bins_theta)
        np.add.at(rho, k,                    w1)
        np.add.at(rho, (k + 1) % bins_theta, w2)
        rho = comm.allreduce(rho, op=MPI.SUM)
        rho /= max(rho.sum() * delta, 1e-30)
        return rho

    def vlasov_force(angle, X, density):  # X, density unused — global mean field
        rho = cic_density_global(angle)
        W_grid = delta * (W_mat @ rho)
        k_idx = (np.floor(angle.ravel() / delta).astype(int)) % bins_theta
        force = L_rod ** 2 * (W_grid[k_idx] - W_grid[(k_idx + 1) % bins_theta]) / delta
        return force.reshape(-1, 1)

    def interaction_energy(angle):
        rho = cic_density_global(angle)
        return float(np.sum(W_mat * rho[:, None] * rho[None, :]) * delta ** 2)

    return vlasov_force, interaction_energy


info = {
    "mass": 1.0,
    "inertia": 1.0,
    "length": L,
    "ev": 1.0,
    "om": 1.0,
    "cutoff": 0.1,
    "cross_section": "hard_needle",       # Onsager-form NTC
    "collision_kind": "enskog",           # Parsons-Lee + position-aware
    "xmin": 0.0, "xmax": Lx,
    "ymin": 0.0, "ymax": Lx,
    "bcs": "periodic",
    "initial_angle_amplitude": 0.1,
    "initial_angle_shift": 0.0,
    "initial_angle_wavelength": 2,        # cos(2θ) seed → director along x̂
}

# Smectic wavevectors: 5 around the natural smectic mode along x̂ and ŷ.
k0 = 2.0 * np.pi / Lx
m_centre = n_layers
m_window = (-2, -1, 0, 1, 2)
smectic_k = (
    [(m_centre + dm) * k0 * np.array([1.0, 0.0]) for dm in m_window]
  + [(m_centre + dm) * k0 * np.array([0.0, 1.0]) for dm in m_window]
)
smectic_k = [tuple(k) for k in smectic_k]

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
    "smectic_k": smectic_k,
    "T_bath": T_bath,
    "nu_bath": nu_bath,
    "prefix": "output/test_needle_smectic_2d",
}

Print("Running needle 2-D smectic test (Onsager + Enskog):")
Print(f"  L={L:.4f},  Lx={Lx:.2f}  ({n_layers} rod-lengths across)")
Print(f"  η_target={eta_target},  nlocal_total≈{N_total},  nlocal/rank={nlocal}")
Print(f"  T_bath={T_bath} ({'< T_NI=2.55, smectic regime' if T_bath < 2.55 else '> T_NI'}),"
      f" nu_bath={nu_bath}, nsteps={nsteps}")
Print(f"  vlasov={'ON (Onsager mean-field)' if use_vlasov else 'OFF (pure Enskog)'},"
      f" smectic_k registered at m∈{[m_centre+dm for dm in m_window]} along x̂ and ŷ")

# Build Vlasov closures if requested.
if use_vlasov:
    vf, ie = _onsager_vlasov_inhomo(L_rod=L, bins_theta=128, comm=MPI.COMM_WORLD)
else:
    vf, ie = None, None

sim = CFMZNeedleDSMC(
    opts=opts, info=info,
    vlasov_force=vf,
    interaction_energy=ie,
    comm=MPI.COMM_WORLD,
)

# Run with periodic VTK output.
vtk_every = 50
n_full_blocks, leftover = divmod(nsteps, vtk_every)
done = 0
for b in range(n_full_blocks):
    sim.run(nsteps=vtk_every, monitor_every=vtk_every)
    done += vtk_every
    sim.export_cell_fields_vtk(
        prefix=f"{sim.output_path}/dsmc_{done:05d}",
        smectic_k=smectic_k,
        time=done * dt,
    )
if leftover:
    sim.run(nsteps=leftover, monitor_every=leftover)
    done += leftover
    sim.export_cell_fields_vtk(
        prefix=f"{sim.output_path}/dsmc_{done:05d}",
        smectic_k=smectic_k,
        time=done * dt,
    )

# Tail-averaged sanity report.
tail = max(1, nsteps // 3)
final = {n: float(np.mean(1.0 - np.array(sim.history[f"circular_var_n{n}"][-tail:])))
         for n in (1, 2, 4, 6)}
psi_x = [float(np.mean(np.array(sim.history[f"smectic_abs_{idx}"][-tail:])))
         for idx in range(5)]
psi_y = [float(np.mean(np.array(sim.history[f"smectic_abs_{idx}"][-tail:])))
         for idx in range(5, 10)]
N_global = nlocal * MPI.COMM_WORLD.Get_size()
noise_floor = 1.0 / np.sqrt(N_global)

if MPI.COMM_WORLD.Get_rank() == 0:
    Print("\nFinal-third averages:")
    Print(f"  R₁={final[1]:.3f}  R₂={final[2]:.3f}  R₄={final[4]:.3f}  R₆={final[6]:.3f}")
    Print(f"  Smectic projections (x̂):")
    for dm, val in zip(m_window, psi_x):
        Print(f"    m={m_centre + dm:>3d}:  ψ_S = {val:.4f}")
    Print(f"  Smectic projections (ŷ):")
    for dm, val in zip(m_window, psi_y):
        Print(f"    m={m_centre + dm:>3d}:  ψ_S = {val:.4f}")
    Print(f"  noise floor 1/√N ≈ {noise_floor:.4f}")
    if max(psi_x) < 5 * noise_floor:
        Print("  WARNING: no x̂-aligned smectic mode rose above noise — "
              "smectic-A did not emerge.  Try lower T_bath, longer run, "
              "or different n_layers / η.")
    if max(psi_y) > 5 * noise_floor:
        Print("  WARNING: ŷ-aligned smectic above noise — "
              "phase looks crystalline rather than smectic-A.")
Print("test_needle_smectic_2d complete.")
