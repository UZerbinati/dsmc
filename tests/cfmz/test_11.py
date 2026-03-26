"""
Pure Vlasov test — Onsager potential, no collisions
----------------------------------------------------
Collisions are disabled by setting ``extra_collision=-1`` (default), so
``extra_collision = int(-1 + 1) = 0`` and the collision loop is skipped.
This is the energy-conservation check: in the collisionless limit the
conserved quantity is

    KE_rot / N  +  (L²/2) · E[ρ]  =  const

where KE_rot = ½ I Σ ωᵢ² is the total rotational kinetic energy,
N is the total particle count, L is the rod length, and

    E[ρ] = ∫∫ |sin(θ₁−θ₂)| ρ(θ₁) ρ(θ₂) dθ₁ dθ₂

is the Onsager interaction energy.  Translational KE is decoupled from
the Vlasov force and is independently conserved.

The Vlasov torque on a particle at θ is
    F(θ) = −L² ∫ sign(sin(θ−θ')) cos(θ−θ') ρ(θ') dθ'.
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
Print("Running homogeneous CFMZ needle DSMC with options:")

nlocal = Opt.getReal("nlocal", 5e5)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 128)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu",4)
nsteps = Opt.getInt("nsteps", 4000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = int(Opt.getReal("extra_collision", -1)+1)
monitor_every = Opt.getInt("monitor_every", 100)

Print(f"  nlocal={nlocal}")
Print(f"  nu={nu}")
Print(f"  dt={dt}")
Print(f"  collision ratio is {nu*dt}")
Print(f"  bins={bins}")
Print(f"  nsteps={nsteps}")
Print(f"  seed={seed}")
Print(f"  monitor_every={monitor_every}")
Print(f"  extra_collision={extra_collision}")
Print(f"  collision_type={collision_type}")
Print(f"  grazing_collision={grazing_collision}")

Print("--------------------------------------------------------------------")

#TODO: Fix with correct relation between length mass and inertia
info = {"inertia": 1.0,
        "mass": 1.0,
        "length": np.sqrt(12.0),
        "ev": 1.0,       # translational restitution
        "om": 1.0,       # rotational restitution
        "cutoff": 0.1,   # angular cutoff
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
        fig.savefig(f"output/test_11_output_cfmz_{collision_type}/vlasov_energy.pdf")
        fig.savefig(f"output/test_11_output_cfmz_{collision_type}/vlasov_energy.png", dpi=400)
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
    "prefix": "output/test_11",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    vlasov_force=vlasov_force,
    interaction_energy=interaction_energy_fn,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")

