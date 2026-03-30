"""
Hard-needle cross-section — Onsager Vlasov potential, ν=0.5
------------------------------------------------------------
Same as test_9 (Onsager mean-field potential W(θ₁,θ₂) = |sin(θ₁−θ₂)|, low
collision frequency ν=0.5) but using the hard-needle NTC kernel instead of the
Maxwell (uniform) kernel.

The Vlasov torque on a particle at θ is
    F(θ) = −L² ∫ sign(sin(θ−θ')) cos(θ−θ') ρ(θ') dθ'
and the interaction energy tracked by the library is
    E[ρ] = ∫∫ |sin(θ₁−θ₂)| ρ(θ₁) ρ(θ₂) dθ₁ dθ₂.

Bird's NTC acceptance–rejection method is applied:
  - Mcol_cand = floor(ν_max · N · dt / 2) candidate pairs are drawn each step.
  - Each candidate is accepted with probability |g·n| · L|sin(Δθ)| / ν_max.
  - The running maximum ν_max (stored as sim._nu_max) is updated every step.

Compare with test_9 (same parameters, Maxwell kernel) to observe the effect of
the anisotropic cross-section in the low-collision regime.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print
import numpy as np
import matplotlib.pyplot as plt
from dsmc.utils import fig_axes

Opt = PETSc.Options()
Print("Running homogeneous CFMZ needle DSMC — hard-needle cross-section (NTC), Onsager Vlasov, ν=0.5:")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 128)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 0.5)
nsteps = Opt.getInt("nsteps", 1000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
monitor_every = Opt.getInt("monitor_every", 100)

Print(f"  nlocal={nlocal}")
Print(f"  nu={nu}  (initial NTC estimate; adapts to max kernel each step)")
Print(f"  dt={dt}")
Print(f"  collision ratio is {nu*dt}")
Print(f"  bins={bins}")
Print(f"  nsteps={nsteps}")
Print(f"  seed={seed}")
Print(f"  monitor_every={monitor_every}")
Print(f"  extra_collision={extra_collision}")
Print(f"  collision_type={collision_type}")
Print(f"  cross_section=hard_needle")
Print(f"  grazing_collision={grazing_collision}")

Print("--------------------------------------------------------------------")

info = {"inertia": 1.0,
        "mass": 1.0,
        "length": np.sqrt(12.0),
        "ev": 1.0,             # translational restitution
        "om": 1.0,             # rotational restitution
        "cutoff": 0.1,         # angular cutoff for near-parallel fallback
        "cross_section": "hard_needle",  # Onsager NTC kernel (arXiv:2508.10744 Example B)
       }
vlasov_energy_history = []

# Precompute kernel matrix K[k,j] = sign(sin(θ_k − θ_j)) cos(θ_k − θ_j)
# on a uniform grid covering [0, 2π).  Computed once; reused every step.
_grid_centers = (np.arange(bins) + 0.5) * (2*np.pi / bins)
_diff  = _grid_centers[:, None] - _grid_centers[None, :]   # (bins, bins)
_K_mat = np.sign(np.sin(_diff)) * np.cos(_diff)             # (bins, bins)

comm = MPI.COMM_WORLD
def vlasov_force(theta):
    hist, theta_edges = np.histogram(theta.ravel(), bins=bins, range=(0.0, 2*np.pi))
    hist = comm.allreduce(hist, op=MPI.SUM)
    delta_theta = 2*np.pi / bins
    rho = hist / (np.sum(hist) * delta_theta)
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    # O(bins²) grid convolution — no O(N×bins) broadcast
    F_grid = -delta_theta * (_K_mat @ rho)                  # (bins,)
    L = info["length"]
    # Interpolate to particle positions: O(N log bins)
    force = L**2 * np.interp(theta.ravel(), centers, F_grid, period=2*np.pi)
    if comm.Get_rank() == 0:
        vlasov_energy_history.append(np.sqrt(np.sum(F_grid**2) * delta_theta))
        fig, ax, _ = fig_axes()
        time = np.array(range(len(vlasov_energy_history)))*dt
        ax.plot(time, vlasov_energy_history, color="black", linewidth=1.5)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\mathcal{V}(\theta)|$")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"output/test_22_output_cfmz_{collision_type}/vlasov_energy.pdf")
        fig.savefig(f"output/test_22_output_cfmz_{collision_type}/vlasov_energy.png", dpi=400)
        plt.close(fig)
    return force.reshape(-1, 1)

def interaction_energy_fn(theta):
    """E[ρ] = ∫∫ |sin(θ₁−θ₂)| ρ(θ₁)ρ(θ₂) dθ₁dθ₂ via histogram quadrature."""
    hist, _ = np.histogram(theta.ravel(), bins=bins, range=(0.0, 2*np.pi))
    hist = comm.allreduce(hist, op=MPI.SUM)
    delta_theta = 2*np.pi / bins
    rho = hist / (np.sum(hist) * delta_theta)
    centers = (np.arange(bins) + 0.5) * delta_theta
    W = np.abs(np.sin(centers[:, None] - centers[None, :]))
    return float(np.sum(W * rho[:, None] * rho[None, :]) * delta_theta**2)

opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": extra_collision,
    "grazing_collision": grazing_collision,
    "collision_type": collision_type,
    "seed": seed,
    "test": "uniform_angle",
    "variance": "real_projective_plane",
    "prefix": "output/test_22",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    vlasov_force=vlasov_force,
    interaction_energy=interaction_energy_fn,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print(f"Simulation complete.  Final NTC running maximum: nu_max={sim._nu_max:.4e}")
