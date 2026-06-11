"""
Test needle smectic 2-D — T_bath phase-diagram sweep
=====================================================

Sweeps the Andersen bath temperature across the I → N → Sm-A
sequence in the 2-D Onsager-plus-Enskog needle system.  Same model
as ``test_needle_smectic_2d.py`` (Onsager Vlasov mean-field +
Enskog Parsons-Lee correction + Andersen thermostat); only T_bath
varies.

For each temperature a fresh ``CFMZNeedleDSMC`` simulation is run and
the steady-state nematic R₂ and smectic order parameters
``ψ_S(m·k₀ x̂)`` and ``ψ_S(m·k₀ ŷ)`` are recorded for a fan of
wavevectors centred on the natural smectic mode ``m = n_layers``.

Phase identification:

| Phase | R₂ | ψ_S(x̂)_max | ψ_S(ŷ)_max |
|---|---|---|---|
| Isotropic | ≈ 0 | ≈ 0 | ≈ 0 |
| Nematic | > 0 | ≈ 0 | ≈ 0 |
| Smectic-A | > 0 | > 0 | ≈ 0 |
| 2-D crystal | > 0 | > 0 | > 0 |

The "≈ 0" baseline is the noise floor 1/√N_global.

Parameter philosophy: same as ``test_needle_smectic_2d`` — physical
units with L = √12 (so T_NI = 8/π ≈ 2.55), box of n_layers
rod-lengths, density chosen to give η ≈ 0.5 (Parsons-Lee active).

T_bath schedule: ``[5.0, 3.5, 2.5, 2.0, 1.5, 1.0, 0.5]`` — crosses
the Onsager spinodal T_NI = 2.55.  Below T_NI we're in the nematic
regime; below some lower T_NA the smectic forms (the precise T_NA is
*measured* from the sweep, not predicted, since it depends on the
Enskog density correction).

Output
------
- ``output/test_needle_smectic_2d_sweep/T_<T>_output_cfmz_inhomo_nanbu/``
  per-T directories.
- ``output/test_needle_smectic_2d_sweep/phase_diagram.{pdf,png}``
  overlaying R₂(T), max_m ψ_S(x̂)(T), max_m ψ_S(ŷ)(T) with a
  horizontal noise-floor guide.
- One VTK snapshot per T_bath at the final step.

CLI flags
---------
- ``-vlasov 1`` (default): Onsager mean-field on; gives the standard
  I-N transition at T_NI ≈ 2.55.
- ``-vlasov 0``: pure Enskog; tests whether the Bates-Frenkel
  smectic-from-hard-rods result transfers to DSMC.
"""
import sys
import os
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from dsmc import CFMZNeedleDSMC, Print
from dsmc.utils import fig_axes

Opt = PETSc.Options()
n_layers = Opt.getInt("n_layers", 50)
eta_target = Opt.getReal("eta", 0.5)
bins = Opt.getInt("bins", 8)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 20.0)
nu_bath = Opt.getReal("nu_bath", 16.0)
nsteps = Opt.getInt("nsteps", 3000)
seed = Opt.getInt("seed", 47)
use_vlasov = bool(Opt.getInt("vlasov", 1))
# de Gennes–McMillan smectic-A drive (see CFMZ.md §14.13).  Off by default;
# enable across the entire T sweep with ``-smectic_drive 1``.  When on, each
# T_bath run gets the same coupling c.
smectic_drive    = bool(Opt.getInt("smectic_drive", 0))
smectic_coupling = Opt.getReal("smectic_coupling", 1.0)

L = np.sqrt(12.0)
Lx = float(n_layers) * L
# Physical particle count from η_target (sets the Parsons-Lee correction).
N_phys = max(int(round((4.0 / np.pi) * eta_target * n_layers ** 2)), 100)
size = max(1, MPI.COMM_WORLD.Get_size())
# -nlocal overrides the η-derived count so statistics scale independently
# of physics; defaults to N_phys // size (original behaviour).
nlocal = max(Opt.getInt("nlocal", max(N_phys // size, 100)), 1)
N_sim_total = nlocal * size
particle_weight = float(N_phys) / float(max(N_sim_total, 1))


def _onsager_vlasov_inhomo(L_rod, bins_theta, comm):
    """Onsager Vlasov closure (calamitic, |sin Δθ|) adapted to the
    inhomogeneous-solver signature.  Uses a global CIC density
    allreduced across ranks.  See ``test_needle_smectic_2d.py`` for
    the docstring.
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

    def vlasov_force(angle, X, density):
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

    Mirrors the closure in ``test_needle_smectic_2d.py``.  Returns
    ``(translational_force, smectic_torque)`` callables that match the
    ``(angle, X, density)`` interface of ``vlasov_kick_step``.  The
    sweep relies on the cos(2θ) IC pinning the director along x̂; the
    drive is therefore expressed in that fixed frame so the smectic
    potential is commensurate with the periodic box.  See CFMZ.md
    §14.13 for the physics.
    """
    k_S = 2.0 * np.pi * n_layers_ / Lx_

    def translational_force(angle, X, density):
        x0 = X[:, 0]
        th = angle.ravel()
        amp = -c_coupling * k_S * np.cos(2.0 * th) * np.sin(k_S * x0)
        out = np.zeros((th.size, 2))
        out[:, 0] = amp
        return out

    def smectic_torque(angle, X, density):
        x0 = X[:, 0]
        th = angle.ravel()
        torque = -2.0 * c_coupling * np.cos(k_S * x0) * np.sin(2.0 * th)
        return torque[:, None]

    return translational_force, smectic_torque


# Wavevector fan: 5 around the natural smectic mode along x̂ and ŷ.
k0 = 2.0 * np.pi / Lx
m_centre = n_layers
m_window = (-2, -1, 0, 1, 2)
smectic_k = (
    [(m_centre + dm) * k0 * np.array([1.0, 0.0]) for dm in m_window]
  + [(m_centre + dm) * k0 * np.array([0.0, 1.0]) for dm in m_window]
)
smectic_k = [tuple(k) for k in smectic_k]

T_bath_values = [5.0, 3.5, 2.5, 2.0, 1.5, 1.0, 0.5]
output_root = "output/test_needle_smectic_2d_sweep"

info_template = {
    "mass": 1.0,
    "inertia": 1.0,
    "length": L,
    "ev": 1.0,
    "om": 1.0,
    "cutoff": 0.1,
    "cross_section": "hard_needle",
    "collision_kind": "enskog",
    "xmin": 0.0, "xmax": Lx,
    "ymin": 0.0, "ymax": Lx,
    "bcs": "periodic",
    "initial_angle_amplitude": 0.1,
    "initial_angle_shift": 0.0,
    "initial_angle_wavelength": 2,
    # Decouple sim sample count from physical density (CFMZ.md §14.10).
    "particle_weight": particle_weight,
    # Cross-section floor (CFMZ.md §14.13): keeps the Enskog kernel
    # collisional even when the Onsager Vlasov pins R₂ → 1 in deep
    # nematic regions of the sweep.
    "sin_dtheta_floor": float(np.sin(0.1)),
}

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(output_root, exist_ok=True)
comm.Barrier()

Print("Running needle 2-D smectic phase-diagram sweep (Onsager + Enskog):")
Print(f"  L={L:.4f},  Lx={Lx:.2f}  ({n_layers} rod-lengths across)")
Print(f"  η_target={eta_target},  N_phys={N_phys},  N_sim={N_sim_total} "
      f"(nlocal/rank={nlocal}, F_N={particle_weight:.3e})")
Print(f"  T_NI (Onsager spinodal) ≈ {8 / np.pi:.3f},  T_bath sweep = {T_bath_values}")
Print(f"  vlasov={'ON' if use_vlasov else 'OFF (pure Enskog)'}")
Print(f"  smectic_drive={'ON' if smectic_drive else 'OFF'}"
      f"{f' (c={smectic_coupling}, k_S=2π/L)' if smectic_drive else ''}")

R2_curve = []
psi_x_max = []
psi_y_max = []

if __name__ == "__main__":
    for T_b in T_bath_values:
        Print(f"\n--- T_bath = {T_b:.2f} ---")
        info = dict(info_template)
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
            "T_bath": T_b,
            "nu_bath": nu_bath,
            "prefix": f"{output_root}/T_{T_b:.2f}",
        }
        if use_vlasov:
            onsager_vf, ie = _onsager_vlasov_inhomo(L_rod=L, bins_theta=128, comm=comm)
        else:
            onsager_vf, ie = None, None

        # de Gennes–McMillan smectic drive composes additively with Onsager.
        if smectic_drive:
            sm_tf, sm_torque = _smectic_drive_inhomo(
                L_rod=L, c_coupling=smectic_coupling,
                n_layers_=n_layers, Lx_=Lx,
            )
            if onsager_vf is not None:
                def vf(angle, X, density,
                       _o=onsager_vf, _s=sm_torque):
                    return _o(angle, X, density) + _s(angle, X, density)
            else:
                vf = sm_torque
            tf = sm_tf
        else:
            vf = onsager_vf
            tf = None

        sim = CFMZNeedleDSMC(opts=opts, info=info,
                             vlasov_force=vf,
                             translational_force=tf,
                             interaction_energy=ie,
                             comm=comm)
        sim.run(nsteps=nsteps, monitor_every=nsteps)
        sim.export_cell_fields_vtk(
            prefix=f"{output_root}/snap_T_{T_b:.2f}",
            smectic_k=smectic_k,
            time=nsteps * dt,
        )

        # Tail-averaged R_2 and the maxima of ψ_S over the registered
        # wavevector fans (x̂- and ŷ-aligned subsets).
        tail = max(1, nsteps // 10)
        R2 = float(np.mean(1.0 - np.array(sim.history["circular_var_n2"][-tail:])))
        psi_x_vals = [
            float(np.mean(np.array(sim.history[f"smectic_abs_{idx}"][-tail:])))
            for idx in range(5)
        ]
        psi_y_vals = [
            float(np.mean(np.array(sim.history[f"smectic_abs_{idx}"][-tail:])))
            for idx in range(5, 10)
        ]
        R2_curve.append(R2)
        psi_x_max.append(max(psi_x_vals))
        psi_y_max.append(max(psi_y_vals))
        Print(f"   R₂={R2:.3f}  max ψ_S(x̂)={psi_x_max[-1]:.4f}  "
              f"max ψ_S(ŷ)={psi_y_max[-1]:.4f}")

    if comm.Get_rank() == 0:
        Print("\nSweep complete — writing phase diagram.")
        # Noise on global ψ_S scales as 1/√N_sim (sample count), not N_phys.
        noise_floor = 1.0 / np.sqrt(N_sim_total)
        fig, ax, _ = fig_axes()
        ax.plot(T_bath_values, R2_curve, "o-", color="C0", linewidth=1.5,
                label=r"$R_2$ (nematic)")
        ax.plot(T_bath_values, psi_x_max, "s-", color="C2", linewidth=1.5,
                label=r"$\max_m \psi_S(m\hat{x})$ (smectic-A signal)")
        ax.plot(T_bath_values, psi_y_max, "d--", color="C3", linewidth=1.2,
                label=r"$\max_m \psi_S(m\hat{y})$ (perpendicular — should stay flat)")
        ax.axhline(noise_floor, color="grey", linestyle=":", linewidth=0.8,
                   label=fr"noise floor $\sim 1/\sqrt{{N}} = {noise_floor:.4f}$")
        ax.axvline(8.0 / np.pi, color="k", linestyle=":", linewidth=0.9,
                   label=fr"$T_{{NI}} = 8/\pi \approx {8/np.pi:.3f}$")
        ax.set_xlabel(r"$T_{\mathrm{bath}}$")
        ax.set_ylabel(r"order parameters")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        title = "Onsager + Enskog" if use_vlasov else "pure Enskog (no Vlasov)"
        ax.set_title(f"Smectic phase diagram — {title}", fontsize=10)
        fig.savefig(f"{output_root}/phase_diagram.pdf")
        fig.savefig(f"{output_root}/phase_diagram.png", dpi=400)
        plt.close(fig)
        Print(f"  Wrote {output_root}/phase_diagram.pdf, .png")
        Print(f"  Wrote per-T VTK snapshots {output_root}/snap_T_*.vtr")
