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

Parameter
----------
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
- IC near-isotropic (``initial_angle_amplitude = 0.01``) so the
  director arises from spontaneous symmetry breaking.  The angular
  fan of registered wavevectors (below) catches the layered density
  wave at whatever angle Sm-A picks.
- **NVE by default** (``nu_bath = 0``).  The Andersen thermostat is
  effectively disabled so the velocity correlations the Enskog
  kernel imprints in each collision survive into the next transport
  step — that is what builds the position-orientation coupling
  needed for Sm-A.  ``T_bath`` still seeds the Maxwell-Boltzmann IC.
  Caveat: under NVE the *effective* temperature drifts upward as
  Vlasov drives nematic order (orientational PE → translational KE);
  if you want a thermostat anyway, pass ``-nu_bath 16``.
- ``info["collision_kind"] = "enskog"`` — Parsons-Lee correction +
  position-derived contact normal.
- Vlasov force: ``calamitic Onsager closure`` (built inline; see
  ``test_needle_12.py`` for the homogeneous version).  Enable with
  ``-vlasov 1`` (default), disable with ``-vlasov 0`` to test the
  pure-Enskog limit.

Smectic wavevectors
-------------------
A 2-D angular fan is registered: ``n_angle`` directions × ``len(m_window)``
magnitudes around the natural smectic mode ``m = n_layers``.  In a
true smectic-A phase, the maximum over angles at the natural magnitude
rises sharply above noise while neighbouring magnitudes stay flat;
the angle index reports the spontaneous director.

VTK output every 50 steps to
``output/test_needle_smectic_2d_output_cfmz_inhomo_nanbu/dsmc_*.vtr``.
The visualisation grid (``vis_bins``) is finer than the Enskog
kernel grid so smectic stripes (period λ = L) are resolved
(Nyquist needs ``dx_vis ≤ L/2``).
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
eta_target = Opt.getReal("eta", 0.75)        # Parsons-Lee active, not capped (CFMZ.md §14.5).
bins = Opt.getInt("bins", 8)                # cell width Lx/8 ≈ 21.7 ≫ L (CFMZ.md §14.5).
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 20.0)
# Default nu_bath = 0 → Andersen thermostat is a no-op (NVE).  T_bath
# still controls the IC seeding (Maxwell-Boltzmann at this temperature)
# but the run-loop ``andersen_thermostat_step`` resamples 0 % of
# particles each step, so post-collision velocity correlations survive.
# Pass ``-nu_bath 16`` from the command line to re-enable canonical-T
# behaviour.
nu_bath = Opt.getReal("nu_bath", 0.0)
T_bath = Opt.getReal("T_bath", 0.1)
nsteps = Opt.getInt("nsteps", 10000)        # ≥ 2× thermal box-traversal time.
seed = Opt.getInt("seed", 47)
use_vlasov = bool(Opt.getInt("vlasov", 1))   # 1 = on (default), 0 = pure Enskog
# Sm-A stability test: start from a pre-formed layered IC to check whether
# the kinetic kernel preserves smectic order (vs. spontaneously emerging
# from uniform).  Off by default.
smectic_ic = bool(Opt.getInt("smectic_ic", 0))
smectic_amp = Opt.getReal("smectic_amp", 0.5)
# de Gennes–McMillan smectic-A drive: adds a soft Vlasov potential that
# couples nematic order to a layered density wave along x̂ at period L.
# Off by default; enable with ``-smectic_drive 1``.  ``smectic_coupling``
# is the dimensionless prefactor c in V_sm = -c·cos(2θ)·cos(k_S·x).
smectic_drive    = bool(Opt.getInt("smectic_drive", 0))
smectic_coupling = Opt.getReal("smectic_coupling", 1.0)

# Physical-units choices.
L = np.sqrt(12.0)                       # rod length — sets T_NI = (2/3π)L² = 8/π ≈ 2.55.
Lx = float(n_layers) * L                # box: n_layers rod-lengths across
# Physical particle count from η_target = (π/4) ρ L² with ρ = N/Lx²:
#   N_phys = (4/π) η · Lx²/L² = (4/π) eta * n_layers²
N_phys = max(int(round((4.0 / np.pi) * eta_target * n_layers ** 2)), 100)
size = max(1, MPI.COMM_WORLD.Get_size())
# Simulator-particle count: -nlocal overrides the η-derived count so the
# user can scale up statistics independently of the physics.  Falls back
# to N_phys // size (the original behaviour) when -nlocal is not set.
nlocal = max(Opt.getInt("nlocal", max(N_phys // size, 100)), 1)
N_sim_total = nlocal * size
# Particle-weight factor: each simulator particle represents F_N physical
# particles, so the Enskog kernel sees η = η_target regardless of N_sim.
particle_weight = float(N_phys) / float(max(N_sim_total, 1))


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


def _smectic_drive_inhomo(L_rod, c_coupling, n_layers_, Lx_):
    """de Gennes–McMillan smectic-A drive, director fixed along x̂.

    Smectic potential: V_sm = -c·cos(2θ)·cos(k_S·x), with k_S = 2π/L
    so layers stack along x̂ with period L (one rod length).  This is
    the kinetic translation of the F ∝ -γ|ψ|²S² coupling in the
    de Gennes–McMillan free energy.

    v1 fixes the director along x̂ to keep the smectic potential
    commensurate with the periodic box.  A dynamic-θ_dir version is a
    follow-up.

    Returns (translational_force, smectic_torque) closures matching the
    (angle, X, density) → (nlocal, …) interface used by
    vlasov_kick_step in transport_inhomo.py.
    """
    k_S = 2.0 * np.pi * n_layers_ / Lx_

    def translational_force(angle, X, density):
        x0 = X[:, 0]
        th = angle.ravel()
        amp = -c_coupling * k_S * np.cos(2.0 * th) * np.sin(k_S * x0)
        out = np.zeros((th.size, 2))
        out[:, 0] = amp           # force along x̂; y-component is zero
        return out

    def smectic_torque(angle, X, density):
        x0 = X[:, 0]
        th = angle.ravel()
        torque = -2.0 * c_coupling * np.cos(k_S * x0) * np.sin(2.0 * th)
        return torque[:, None]

    return translational_force, smectic_torque


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
    "initial_angle_amplitude": 0.01,      # near-isotropic seed
    "initial_angle_shift": 0.0,
    "initial_angle_wavelength": 2,        # cos(2θ) IC bias
    # Decouple sim sample count from physical density (CFMZ.md §14.10).
    "particle_weight": particle_weight,
    # Cross-section floor: when Vlasov pins R₂→1, |sin Δθ| collapses
    # and the Enskog kernel loses its collisions, killing the position
    # -orientation coupling that drives Sm-A.  Floor at sin(cutoff) so
    # parallel rods still collide as if separated by ≈ cutoff radians
    # (the same angle below which the spherical impulse takes over).
    "sin_dtheta_floor": float(np.sin(0.1)),
    # Smectic IC parameters (only consulted when -smectic_ic 1).
    "smectic_amplitude": smectic_amp,
    "smectic_n_layers": n_layers,
}

# Smectic wavevector fan: n_angle directions × len(m_window) magnitudes
# around the natural smectic mode m = n_layers.  With near-isotropic
# IC the spontaneous director picks a random angle, so we cover [0, π)
# (k and −k give the same |ψ_S|) at 15° resolution to catch it.
k0 = 2.0 * np.pi / Lx
m_centre = n_layers
m_window = (-2, -1, 0, 1, 2)
n_magnitude = len(m_window)
n_angle = 12
angles = np.linspace(0.0, np.pi, n_angle, endpoint=False)
smectic_k = []
for theta in angles:
    direction = np.array([np.cos(theta), np.sin(theta)])
    for dm in m_window:
        smectic_k.append(tuple((m_centre + dm) * k0 * direction))

opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": 1,
    "collision_type": "nanbu",
    "seed": seed,
    "test": "smectic_2d" if smectic_ic else "uniform_perturbed_2d",
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
Print(f"  eta_target={eta_target},  N_phys={N_phys},  N_sim={N_sim_total} "
      f"(nlocal/rank={nlocal}, F_N={particle_weight:.3e})")
thermo_mode = "NVE (no thermostat)" if nu_bath == 0.0 else f"Andersen (nu_bath={nu_bath})"
Print(f"  T_init={T_bath} ({'< T_NI=2.55, smectic regime' if T_bath < 2.55 else '> T_NI'}),"
      f" {thermo_mode}, nsteps={nsteps}")
Print(f"  vlasov={'ON (Onsager mean-field)' if use_vlasov else 'OFF (pure Enskog)'},"
      f" smectic_k fan = {n_angle} angles × {n_magnitude} magnitudes "
      f"(m∈{[m_centre+dm for dm in m_window]})")
Print(f"  smectic_drive={'ON' if smectic_drive else 'OFF'}"
      f"{f' (c={smectic_coupling}, k_S=2π/L)' if smectic_drive else ''}")
Print(f"  bins (Enskog kernel) = {bins}, dx_kernel = Lx/bins ≈ {Lx/bins:.2f}")
if Opt.getInt("nlocal", -1) <= 0:
    Print("  Tip: pass `-nlocal 100000` (or 2_500_000 for production) "
          "for cleaner statistics; this run uses N_phys particles.")

# Build Vlasov closures if requested.
if use_vlasov:
    onsager_vf, ie = _onsager_vlasov_inhomo(L_rod=L, bins_theta=256, comm=MPI.COMM_WORLD)
else:
    onsager_vf, ie = None, None

# de Gennes–McMillan smectic drive: composes a translational force and an
# extra angular torque on top of the Onsager mean field.  Total torque =
# Onsager + smectic when both are active.
if smectic_drive:
    sm_tf, sm_torque = _smectic_drive_inhomo(L_rod=L, c_coupling=smectic_coupling,
                                             n_layers_=n_layers, Lx_=Lx)
    if onsager_vf is not None:
        def vf(angle, X, density):
            return onsager_vf(angle, X, density) + sm_torque(angle, X, density)
    else:
        vf = sm_torque
    tf = sm_tf
else:
    vf = onsager_vf
    tf = None

sim = CFMZNeedleDSMC(
    opts=opts, info=info,
    vlasov_force=vf,
    translational_force=tf,
    interaction_energy=ie,
    comm=MPI.COMM_WORLD,
)

# Run continuously; emit a VTK snapshot every `vtk_every` steps via the
# callback hook on `run`.  The callback fires inside the integration loop,
# so the swarm/RNG/history state is *not* perturbed between snapshots.
vtk_every = 50

# Visualisation grid finer than the Enskog kernel grid so the smectic
# layers (period λ = L) are resolved (to avoid aliasing we need dx_vis ≤ L/2;
# we go# ~4× finer).
# This Affects the VTK output only — the kernel still uses `bins`.
vis_bins = max(8 * n_layers, 4 * bins)

def _vtk_snapshot(step):
    sim.export_cell_fields_vtk(
        prefix=f"{sim.output_path}/dsmc_{step:05d}",
        time=step * dt,
        vis_bins=vis_bins,
    )
    # Refresh the PVD index after every snapshot so the time series is
    # navigable in ParaView while the simulation is still running.
    sim.write_pvd_collection(f"{sim.output_path}/dsmc.pvd")

sim.run(nsteps=nsteps, monitor_every=vtk_every,
        callback=_vtk_snapshot, callback_every=vtk_every)

# Tail-averaged sanity report.  Reshape the (angle, magnitude) fan
# stored in sim.history and take the max over angles per magnitude:
# in Sm-A only one (m, θ) bin should rise above the 1/√N_sim noise
# floor, and that θ is the spontaneous director.
tail = max(1, nsteps // 3)
final_R = {n: float(np.mean(1.0 - np.array(sim.history[f"circular_var_n{n}"][-tail:])))
           for n in (1, 2, 4, 6)}
psi_grid = np.array([
    np.mean(np.array(sim.history[f"smectic_abs_{a*n_magnitude+i}"][-tail:]))
    for a in range(n_angle) for i in range(n_magnitude)
]).reshape(n_angle, n_magnitude)
psi_max_per_m = psi_grid.max(axis=0)
psi_argmax_a  = psi_grid.argmax(axis=0)
# Statistical noise on global ψ_S scales as 1/√N_sim (the sample count),
# not 1/√N_phys; oversampling (F_N<1) buys lower noise at fixed physics.
noise_floor = 1.0 / np.sqrt(N_sim_total)

if MPI.COMM_WORLD.Get_rank() == 0:
    Print("\nFinal-third averages:")
    Print(f"  R₁={final_R[1]:.3f}  R₂={final_R[2]:.3f}  "
          f"R₄={final_R[4]:.3f}  R₆={final_R[6]:.3f}")
    Print(f"  Smectic max over angles per magnitude:")
    for i, dm in enumerate(m_window):
        m = m_centre + dm
        ang_deg = np.degrees(angles[psi_argmax_a[i]])
        Print(f"    m={m:>3d}:  max_θ ψ_S = {psi_max_per_m[i]:.4f}"
              f"  at θ ≈ {ang_deg:5.1f}°")
    Print(f"  noise floor 1/√N ≈ {noise_floor:.4f}")
    natural = m_window.index(0)
    natural_psi = psi_max_per_m[natural]
    natural_ang = np.degrees(angles[psi_argmax_a[natural]])
    if natural_psi < 5 * noise_floor:
        Print("  WARNING: natural-mode ψ_S did not rise above noise — "
              "smectic-A did not emerge.  Try lower T_bath, longer run, "
              "or larger -nlocal.")
    else:
        Print(f"  ✓ Sm-A signal: ψ_S(m={m_centre}) ≈ {natural_psi:.4f} "
              f"({natural_psi/noise_floor:.1f}×noise) along θ ≈ {natural_ang:.1f}°.")
    # Crystallinity check: are off-natural magnitudes also far above noise?
    other_max = float(np.delete(psi_max_per_m, natural).max())
    if other_max > 0.5 * natural_psi and natural_psi > 5 * noise_floor:
        Print("  WARNING: a second magnitude is comparable to the natural "
              "mode — phase looks crystalline rather than smectic-A.")
Print("test_needle_smectic_2d complete.")
