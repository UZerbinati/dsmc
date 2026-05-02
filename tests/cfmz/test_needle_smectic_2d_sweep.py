"""
Test needle smectic 2-D — T_bath phase-diagram sweep
======================================================

Sweeps the Andersen bath temperature across the I → N → Sm-A
sequence in the 2-D Enskog needle system, and produces a phase
diagram showing all three regimes.  Same model as
``test_needle_smectic_2d`` (Onsager kernel + Parsons-Lee correction +
Andersen thermostat); only T_bath varies.

For each temperature a fresh ``CFMZNeedleDSMC`` simulation is run and
the steady-state nematic R₂ and smectic order parameters
ψ_S(k‖x̂) and ψ_S(k‖ŷ) are recorded.  The phase identification:

  Isotropic  : R₂ ≈ 0,                 ψ_S(x̂) ≈ ψ_S(ŷ) ≈ 0
  Nematic    : R₂ > 0,                 ψ_S(x̂) ≈ ψ_S(ŷ) ≈ 0
  Smectic-A  : R₂ > 0,                 ψ_S(x̂) > 0, ψ_S(ŷ) ≈ 0

The smectic-A vs 2-D-crystal distinction is in the ŷ component:
a crystal would have ψ_S(ŷ) > 0 too.  In a true smectic-A the
in-plane (perpendicular-to-director) direction stays liquid.

Output
------
- ``output/test_needle_smectic_2d_sweep/T_<T>_output_cfmz_inhomo_nanbu/``
  per-T directories with full history.pickle.
- ``output/test_needle_smectic_2d_sweep/phase_diagram.{pdf,png}``
  overlaying R₂(T), ψ_S(x̂)(T), ψ_S(ŷ)(T) with vertical guides at the
  observed transition temperatures.
- One VTK snapshot per T_bath at the final step
  (``snap_<T>.vtr``) for visual inspection in ParaView.

Suggested cost (default parameters)
-----------------------------------
7 T_bath values × 1500 steps × 50 000 particles ≈ 5 × 10⁸ particle-steps.
~ 10 minutes on a single rank for a 2-D simulation; scales linearly
with rank count up to ~ 4 ranks (after which the missing-cross-rank-
ghost limitation of the Enskog kernel starts to matter).
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
nlocal = int(Opt.getReal("nlocal", 5e4))
bins = Opt.getInt("bins", 8)
dt = Opt.getReal("dt", 0.01)
nu = Opt.getReal("nu", 50.0)
nu_bath = Opt.getReal("nu_bath", 16.0)
nsteps = Opt.getInt("nsteps", 1500)
seed = Opt.getInt("seed", 47)

L = 0.10
Lx = 1.0

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
}

# Seven-point T-schedule covering the three regimes.
T_bath_values = [1.5, 1.0, 0.8, 0.6, 0.4, 0.30, 0.15]
output_root = "output/test_needle_smectic_2d_sweep"

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(output_root, exist_ok=True)
comm.Barrier()

Print("Running needle 2-D smectic phase-diagram sweep (Enskog):")
Print(f"  nlocal={nlocal}/rank, L={L}, T_bath in {T_bath_values}")
Print(f"  η_avg ≈ {(np.pi/4) * (nlocal * comm.Get_size()) * L * L:.2f}")

R2_curve = []
psi_x_curve = []
psi_y_curve = []

if __name__ == "__main__":
    for T_b in T_bath_values:
        Print(f"\n--- T_bath = {T_b:.2f} ---")
        info = dict(info_template)
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
            "T_bath": T_b,
            "nu_bath": nu_bath,
            "prefix": f"{output_root}/T_{T_b:.2f}",
        }
        sim = CFMZNeedleDSMC(opts=opts, info=info, comm=comm)
        sim.run(nsteps=nsteps, monitor_every=nsteps)

        # One VTK snapshot per T at the final step.
        sim.export_cell_fields_vtk(
            prefix=f"{output_root}/snap_T_{T_b:.2f}",
            smectic_k=opts["smectic_k"],
            time=nsteps * dt,
        )

        # Tail-averaged R_2 and ψ_S(x̂), ψ_S(ŷ).
        tail = max(1, nsteps // 10)
        R2 = float(np.mean(1.0 - np.array(sim.history["circular_var_n2"][-tail:])))
        psi_x = float(np.mean(np.array(sim.history["smectic_abs_0"][-tail:])))
        psi_y = float(np.mean(np.array(sim.history["smectic_abs_1"][-tail:])))
        R2_curve.append(R2)
        psi_x_curve.append(psi_x)
        psi_y_curve.append(psi_y)
        Print(f"   R₂={R2:.3f}  ψ_S(x̂)={psi_x:.4f}  ψ_S(ŷ)={psi_y:.4f}")

    if comm.Get_rank() == 0:
        Print("\nSweep complete — writing phase diagram.")
        N_global = nlocal * comm.Get_size()
        noise_floor = 1.0 / np.sqrt(N_global)
        fig, ax, _ = fig_axes()
        ax.plot(T_bath_values, R2_curve, "o-", color="C0", linewidth=1.5,
                label=r"$R_2$ (nematic)")
        ax.plot(T_bath_values, psi_x_curve, "s-", color="C2", linewidth=1.5,
                label=r"$\psi_S(k\hat{x})$ (smectic-A signal)")
        ax.plot(T_bath_values, psi_y_curve, "d--", color="C3", linewidth=1.2,
                label=r"$\psi_S(k\hat{y})$ (perpendicular — should stay at noise)")
        ax.axhline(noise_floor, color="grey", linestyle=":", linewidth=0.8,
                   label=fr"noise floor $\sim 1/\sqrt{{N}} = {noise_floor:.4f}$")
        ax.set_xlabel(r"$T_{\mathrm{bath}}$")
        ax.set_ylabel(r"order parameters")
        ax.set_xscale("linear")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"{output_root}/phase_diagram.pdf")
        fig.savefig(f"{output_root}/phase_diagram.png", dpi=400)
        plt.close(fig)
        Print(f"  Wrote {output_root}/phase_diagram.pdf, .png")
        Print(f"  Wrote per-T VTK snapshots {output_root}/snap_T_*.vtr")
