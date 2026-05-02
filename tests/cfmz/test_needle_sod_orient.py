"""
Test needle Sod tube — orientation Riemann
=============================================

A *rod-specific* Riemann problem with no calamitic-fluid analogue:
density and temperature are uniform on both sides of the box, but the
**orientation distribution is discontinuous**.  Rods on the left are
concentrated along x̂ via von-Mises with κ = 4; rods on the right are
concentrated along ŷ.  This tests the orientational diffusion of a
sharp angular gradient under the rod-rod collision operator — a
process that has no analogue in scalar (calamitic-fluid) Sod tubes.

Setup
-----
- 1-D periodic spatial domain $[0, L_x]$, $L_x = 1$, `bins = 8`.
- Uniform density and temperature ($\rho_L = \rho_R = 1.5$;
  $T_L = T_R = 1.0$).
- Left:  θ ~ vonMises(0, 4) → concentrated along x̂.
- Right: θ ~ vonMises(π/2, 4) → concentrated along ŷ.
- ``collision_kind = "enskog"`` so the rod-rod scattering is
  position-aware; the orientation diffusion timescale is
  $\sim 1/(\nu \langle |\sin\Delta\theta|\rangle)$ which scales like
  the inverse Onsager rate.

Pass criteria
-------------
- ρ(x) and T(x) profiles stay flat throughout (no spurious shock from
  the orientation discontinuity).
- The orientation-discontinuity smooths over time: the per-bin
  ⟨cos 2θ⟩(x) profile transitions smoothly from cos(0) = +1 on the
  left to cos(π) = −1 on the right.
- Eventually (long times) ⟨cos 2θ⟩ saturates at the homogeneous
  Onsager equilibrium value for the average density / temperature.

Output
------
- ``output/test_needle_sod_orient_output_cfmz_inhomo_nanbu/``
  with full history and a final-step VTK snapshot.
- ``output/test_needle_sod_orient/orientation_profile.{pdf,png}``
  showing ⟨cos 2θ⟩(x) at three time slices.
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
nsteps = Opt.getInt("nsteps", 300)
seed = Opt.getInt("seed", 47)

L_rod = 0.10
Lx = 1.0
output_root = "output/test_needle_sod_orient"

info = {
    "mass": 1.0,
    "inertia": 1.0,
    "length": L_rod,
    "ev": 1.0,
    "om": 1.0,
    "cutoff": 0.1,
    "cross_section": "hard_needle",
    "collision_kind": "enskog",
    "Lx": Lx,
    "bcs": "reflective",
    # Same density / temperature on both sides — only orientation differs.
    "rho_left":  1.5, "rho_right": 1.5,
    "T_left":  1.0,   "T_right":   1.0,
    # Orientation Riemann.
    "left_mean_angle":  0.0,         # vonMises along x̂ on left
    "left_concentration":  4.0,
    "right_mean_angle": 0.5 * np.pi, # vonMises along ŷ on right
    "right_concentration": 4.0,
}
opts = {
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
    "prefix": output_root,
}

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    os.makedirs(output_root, exist_ok=True)
comm.Barrier()

Print("Running needle orientation-Riemann Sod test:")
Print(f"  ρ uniform, T uniform; θ discontinuous: vonMises(0, 4) | vonMises(π/2, 4)")


def per_cell_cos2(sim):
    """Compute ⟨cos 2θ⟩(x) per spatial cell on rank 0."""
    coord_names = sim.swarm.getCellDMActive().getCoordinateFields()
    pos = sim.swarm.getField(coord_names[0])
    angle = sim.swarm.getField("orientation")
    try:
        X = np.asarray(pos).reshape(sim.nlocal, sim.mesh_dim)
        theta = np.asarray(angle).ravel()
        i_idx = np.clip((X[:, 0] / (Lx / sim.bins)).astype(np.int64),
                        0, sim.bins - 1)
        counts = np.bincount(i_idx, minlength=sim.bins).astype(np.float64)
        sum_cos2 = np.bincount(i_idx, weights=np.cos(2 * theta),
                               minlength=sim.bins)
    finally:
        sim.swarm.restoreField(coord_names[0])
        sim.swarm.restoreField("orientation")
    counts_g = np.zeros_like(counts)
    sum_cos2_g = np.zeros_like(sum_cos2)
    sim.comm.Reduce(counts, counts_g, op=MPI.SUM, root=0)
    sim.comm.Reduce(sum_cos2, sum_cos2_g, op=MPI.SUM, root=0)
    if sim.comm.Get_rank() == 0:
        return np.where(counts_g > 0, sum_cos2_g / np.maximum(counts_g, 1), 0.0)
    return None


sim = CFMZNeedleDSMC(opts=opts, info=info, comm=comm)

# Capture three time-slice snapshots: t=0, t=mid, t=final.
snapshots = {}
snapshots["t=0"] = per_cell_cos2(sim)
sim.run(nsteps=nsteps // 2, monitor_every=nsteps)
snapshots[f"t={(nsteps // 2) * dt:.2f}"] = per_cell_cos2(sim)
sim.run(nsteps=nsteps - nsteps // 2, monitor_every=nsteps)
snapshots[f"t={nsteps * dt:.2f}"] = per_cell_cos2(sim)

sim.export_cell_fields_vtk(prefix=f"{sim.output_path}/snap_final", time=nsteps * dt)

if comm.Get_rank() == 0:
    x_centres = (np.arange(bins) + 0.5) * (Lx / bins)
    fig, ax, _ = fig_axes()
    for label, profile in snapshots.items():
        ax.step(x_centres, profile, where="mid", linewidth=1.5, label=label)
    ax.axvline(0.5 * Lx, color="k", linestyle=":", linewidth=0.8,
               label="initial discontinuity")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\langle\cos 2\theta\rangle(x)$")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(f"{output_root}/orientation_profile.pdf")
    fig.savefig(f"{output_root}/orientation_profile.png", dpi=400)
    plt.close(fig)
    Print(f"  Wrote {output_root}/orientation_profile.{{pdf,png}}")
    Print(f"  Initial profile: +1 (left, along x̂) | -1 (right, along ŷ)")
    Print(f"  Late profile should be smoothed by orientational diffusion.")
Print("test_needle_sod_orient complete.")
