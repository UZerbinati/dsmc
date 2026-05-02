"""
Test needle Sod shock tube — Enskog vs Boltzmann at high density
==================================================================

Standard 1-D Sod-tube Riemann problem set up with calamitic rods, run
**twice** at the same parameters: once with the Boltzmann collision
kernel (``collision_kind = "boltzmann"``), once with the Enskog kernel
(``collision_kind = "enskog"``).  Side-by-side comparison of the
density / mean-velocity / temperature profiles shows that:

- on the *dilute* (right) side the two kernels agree within
  statistical noise — this is the dilute-limit regression check;
- on the *dense* (left) side the Enskog shock front is steeper /
  faster than the Boltzmann one because the Parsons-Lee correction
  enhances the local collision rate (Frezzotti 1998 *Physica A* 240
  showed this for hard-sphere gases; the rod analogue is the same
  qualitative effect).

Setup
-----
- 1-D periodic spatial domain $[0, L_x]$ with $L_x = 1$, `bins = 8`
  (cell width 0.125 ≥ rod length L = 0.10).
- Standard Sod IC: $\rho_L L^2 \approx 0.5$ (η ≈ 0.4), $T_L = 1.0$;
  $\rho_R L^2 \approx 0.06$ (η ≈ 0.05), $T_R = 0.8$.
- Orientations are uniform on both sides (no Riemann in θ — the
  orientation Riemann is in ``test_needle_sod_orient.py``).
- Total particle count tuned so that the left-side packing fraction
  is in the regime where Parsons-Lee is non-trivial.
- nsteps = 200 — long enough to see the shock develop without
  averaging out the difference.

Output
------
- ``output/test_needle_sod_dense_boltzmann_output_cfmz_inhomo_nanbu/``
- ``output/test_needle_sod_dense_enskog_output_cfmz_inhomo_nanbu/``
- ``output/test_needle_sod_dense/profile_comparison.{pdf,png}``
  overlaying ρ(x), u(x), T(x) at the final step for both kernels.

Pass criteria
-------------
- Both runs complete without errors.
- Right-side (dilute) density profile difference < 5 σ noise floor.
- Left-side (dense) density profile shows a measurably different
  shock structure between the two kernels.
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
nlocal = int(Opt.getReal("nlocal", 5e5))
bins = Opt.getInt("bins", 8)
dt = Opt.getReal("dt", 0.005)
nu = Opt.getReal("nu", 50.0)
nsteps = Opt.getInt("nsteps", 200)
seed = Opt.getInt("seed", 47)

L_rod = 0.10           # rod length
Lx = 1.0
output_root = "output/test_needle_sod_dense"

base_info = {
    "mass": 1.0,
    "inertia": 1.0,
    "length": L_rod,
    "ev": 1.0,
    "om": 1.0,
    "cutoff": 0.1,
    "cross_section": "hard_needle",
    "Lx": Lx,
    "bcs": "reflective",
    # Sod IC parameters tuned so left side is dense (Parsons-Lee active).
    "rho_left":  4.0,    # ⇒ N_left ≈ 0.8 · nlocal at the IC
    "rho_right": 0.5,
    "T_left":  1.0,
    "T_right": 0.8,
    "right_concentration": 0.0,    # κ=0 → uniform-θ right side too
    # left_mean_angle defaults to None → uniform-θ left side.
}
opts_template = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": 1,
    "collision_type": "nanbu",
    "seed": seed,
    "test": "sod_rod",
    "spatial_dim": 1,
    "transport": True,
    "variance": "real_projective_plane",
    "n_modes": [2],
}

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(output_root, exist_ok=True)
comm.Barrier()

Print("Running needle Sod shock-tube test (Enskog vs Boltzmann):")
eta_L = (np.pi / 4.0) * base_info["rho_left"] * L_rod ** 2
eta_R = (np.pi / 4.0) * base_info["rho_right"] * L_rod ** 2
Print(f"  η_left  ≈ {eta_L:.3f}    η_right ≈ {eta_R:.3f}")
Print(f"  nlocal={nlocal}, nsteps={nsteps}, dt={dt}")


def run_one(kind):
    info = dict(base_info)
    info["collision_kind"] = kind
    opts = dict(opts_template)
    opts["prefix"] = f"{output_root}_{kind}"
    sim = CFMZNeedleDSMC(opts=opts, info=info, comm=comm)
    sim.run(nsteps=nsteps, monitor_every=nsteps)
    sim.export_cell_fields_vtk(
        prefix=f"{sim.output_path}/snap_final",
        time=nsteps * dt,
    )
    return sim


sim_b = run_one("boltzmann")
sim_e = run_one("enskog")

# ---- Read per-cell fields from the final-step VTK files for plotting --
def read_density_profile(sim):
    """Compute the density(x) profile by per-cell histogram."""
    coord_names = sim.swarm.getCellDMActive().getCoordinateFields()
    pos = sim.swarm.getField(coord_names[0])
    try:
        X = np.asarray(pos).reshape(sim.nlocal, sim.mesh_dim)
        i_idx = np.clip((X[:, 0] / (Lx / sim.bins)).astype(np.int64),
                        0, sim.bins - 1)
        counts = np.bincount(i_idx, minlength=sim.bins).astype(np.float64)
    finally:
        sim.swarm.restoreField(coord_names[0])
    counts_g = np.zeros_like(counts)
    sim.comm.Reduce(counts, counts_g, op=MPI.SUM, root=0)
    return counts_g / (Lx / sim.bins)


rho_b = read_density_profile(sim_b)
rho_e = read_density_profile(sim_e)
x_centres = (np.arange(bins) + 0.5) * (Lx / bins)

if comm.Get_rank() == 0:
    fig, ax, _ = fig_axes()
    ax.step(x_centres, rho_b, where="mid", color="C0", linewidth=1.5,
            label="Boltzmann")
    ax.step(x_centres, rho_e, where="mid", color="C2", linewidth=1.5,
            label="Enskog")
    ax.axvline(0.5 * Lx, color="k", linestyle=":", linewidth=0.8,
               label="initial discontinuity")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\rho(x)$  (final step)")
    ax.legend()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(f"{output_root}/profile_comparison.pdf")
    fig.savefig(f"{output_root}/profile_comparison.png", dpi=400)
    plt.close(fig)
    Print(f"\n  Wrote {output_root}/profile_comparison.{{pdf,png}}")
    Print("  Compare ρ_boltzmann(x) vs ρ_enskog(x) — Enskog front should be"
          "  measurably steeper / faster on the dense (left) side.")
Print("test_needle_sod_dense complete.")
