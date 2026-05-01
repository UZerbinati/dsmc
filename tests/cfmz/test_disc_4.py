"""
Discotic test 4 — phase-diagram sweep R_n vs T_bath
====================================================

Sweep T_bath across the discotic isotropic-tetratic transition for the
2-D mean-field disc-Onsager system with R = √12.  For each T_bath a
fresh ``CFMZDiscDSMCHomo`` simulation is run with an Andersen
thermostat; tail-averaged R₁, R₂, R₄, R₆ are recorded.

Estimated spinodal: T_c = R²·2/(3π) = 8/π ≈ 2.55 (the same numerical
value as the calamitic case, but the unstable mode is cos(4θ) rather
than cos(2θ); see CFMZ.md §13.2).

Output
------
- ``output/test_disc_4/T_<T>_output_cfmz_disc_nanbu/`` per-T directories
- ``output/test_disc_4/phase_diagram.{pdf,png}`` overlaying the four
  R_n curves vs T_bath.
"""
import sys
import os
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from dsmc import CFMZDiscDSMCHomo, Print
from dsmc.utils import fig_axes

Opt = PETSc.Options()
nlocal = int(Opt.getReal("nlocal", 1e5))
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4.0)
nu_bath = Opt.getReal("nu_bath", 4.0)
nsteps = Opt.getInt("nsteps", 1000)
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
    "initial_angle_wavelength": 4,
}

T_bath_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
                 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
output_root = "output/test_disc_4"

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(output_root, exist_ok=True)
comm.Barrier()

Print("Running discotic phase-diagram sweep (test 4):")
Print(f"  R={R:.3f}, T_c ≈ {8/np.pi:.3f}")
Print(f"  nlocal={nlocal} per rank, nsteps={nsteps}, T_bath in {T_bath_values}")

# Per-temperature R_n results. Each list is per-T_bath, tail-averaged.
results = {n: [] for n in (1, 2, 4, 6)}

if __name__ == "__main__":
    for T_b in T_bath_values:
        Print(f"\n--- T_bath = {T_b:.2f}  (α = {R*R/T_b:.2f}) ---")
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
            "T_bath": T_b,
            "nu_bath": nu_bath,
            "prefix": f"{output_root}/T_{T_b:.2f}",
        }
        sim = CFMZDiscDSMCHomo(opts=opts, info=info, comm=comm)
        sim.run(nsteps=nsteps, monitor_every=nsteps)
        # Tail-averaged R_n over the last 10 % of the run.
        tail = max(1, nsteps // 10)
        for n in (1, 2, 4, 6):
            sigma2 = np.array(sim.history[f"circular_var_n{n}"][-tail:])
            results[n].append(float(np.mean(1.0 - sigma2)))
        Print(f"   R₁={results[1][-1]:.3f} R₂={results[2][-1]:.3f} "
              f"R₄={results[4][-1]:.3f} R₆={results[6][-1]:.3f}")

    if comm.Get_rank() == 0:
        Print("\nSweep complete — writing phase diagram.")
        fig, ax, _ = fig_axes()
        styles = {1: ("o-", "C3", r"$R_1$"),
                  2: ("s-", "C0", r"$R_2$"),
                  4: ("^-", "C2", r"$R_4$"),
                  6: ("d-", "C1", r"$R_6$")}
        for n in (1, 2, 4, 6):
            marker, color, label = styles[n]
            ax.plot(T_bath_values, results[n], marker, color=color, label=label,
                    linewidth=1.4, markersize=5)
        ax.axvline(8 / np.pi, color="k", linestyle=":", linewidth=0.9,
                   label=fr"$T_c = 8/\pi \approx {8/np.pi:.3f}$")
        ax.set_xlabel(r"$T_{\mathrm{bath}}$")
        ax.set_ylabel(r"$R_n$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"{output_root}/phase_diagram.pdf")
        fig.savefig(f"{output_root}/phase_diagram.png", dpi=400)
        plt.close(fig)
        Print(f"  Wrote {output_root}/phase_diagram.pdf, .png")
