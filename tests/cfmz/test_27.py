"""
Phase diagram: circular variance vs temperature (Andersen thermostat)
---------------------------------------------------------------------
Sweeps T_bath across the isotropic–nematic transition for the 2-D
mean-field Onsager system with L = √12.  For each temperature a fresh
simulation with nlocal = 1e5 particles (one order of magnitude fewer
than the standard tests) is run for nsteps steps with an Andersen
thermostat.  The steady-state circular variance on the real projective
plane,

    σ² = 1 − |⟨exp(2iθ)⟩|,

is recorded as the nematic order parameter (σ² ≈ 1: isotropic;
σ² ≈ 0: nematic) and plotted against T_bath.

Estimated spinodal temperature: T_c = L² · 2/(3π) = 8/π ≈ 2.55.
The actual first-order transition lies close to this value.
"""
import sys
import os
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print
import numpy as np
import matplotlib.pyplot as plt
from dsmc.utils import fig_axes

Opt = PETSc.Options()
Print("Running phase-diagram sweep — CFMZ needle DSMC with Andersen thermostat:")

nlocal = Opt.getReal("nlocal", 1e5)       # one order of magnitude fewer particles
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4)
nsteps = Opt.getInt("nsteps", 1000)
seed = Opt.getInt("seed", 47)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0) + 1
nu_bath = Opt.getReal("nu_bath", 4.0)

Print(f"  nlocal={nlocal}  (per rank)")
Print(f"  nu={nu},  dt={dt},  nsteps={nsteps}")
Print(f"  nu_bath={nu_bath},  bins={bins},  seed={seed}")
Print(f"  Estimated spinodal T_c = L²·2/(3π) = 8/π ≈ {8/np.pi:.3f}")
Print("--------------------------------------------------------------------")

info = {"inertia": 1.0,
        "mass": 1.0,
        "length": np.sqrt(12.0),
        "ev": 1.0,
        "om": 1.0,
        "cutoff": 0.1,
        "cross_section": "maxwell",
        "initial_angle_amplitude": 1e-1,
        "initial_angle_shift": -0.3,
        "initial_angle_wavelength": 1,
       }

# Kernel matrices — computed once, reused for every temperature.
_grid_centers = (np.arange(bins) + 0.5) * (2*np.pi / bins)
_diff  = _grid_centers[:, None] - _grid_centers[None, :]
_W_mat = np.abs(np.sin(_diff))

comm = MPI.COMM_WORLD
_delta_theta = 2*np.pi / bins
_centers = (np.arange(bins) + 0.5) * _delta_theta

def _cic_density(theta_local):
    t = theta_local.ravel() / _delta_theta
    k = np.floor(t).astype(int) % bins
    w2 = t - np.floor(t)
    w1 = 1.0 - w2
    rho = np.zeros(bins)
    np.add.at(rho, k,              w1)
    np.add.at(rho, (k + 1) % bins, w2)
    rho = comm.allreduce(rho, op=MPI.SUM)
    rho /= (rho.sum() * _delta_theta)
    return rho

def vlasov_force(theta):
    rho = _cic_density(theta)
    W_grid = _delta_theta * (_W_mat @ rho)
    L = info["length"]
    k_idx = (np.floor(theta.ravel() / _delta_theta).astype(int)) % bins
    force = L**2 * (W_grid[k_idx] - W_grid[(k_idx + 1) % bins]) / _delta_theta
    return force.reshape(-1, 1)

def interaction_energy_fn(theta):
    rho = _cic_density(theta)
    W = np.abs(np.sin(_centers[:, None] - _centers[None, :]))
    return float(np.sum(W * rho[:, None] * rho[None, :]) * _delta_theta**2)

# Temperature sweep: dense around T_c ≈ 2.55, coarser away from it.
T_bath_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

output_root = "output/test_27"
if comm.Get_rank() == 0:
    os.makedirs(output_root, exist_ok=True)
comm.Barrier()

circular_vars = []

for T_b in T_bath_values:
    Print(f"\n--- T_bath = {T_b:.2f}  (α = {12.0/T_b:.2f}) ---")
    opts = {
        "nlocal": nlocal,
        "nu": nu,
        "dt": dt,
        "bins": bins,
        "extra_collision": extra_collision,
        "grazing_collision": False,
        "collision_type": collision_type,
        "seed": seed,
        "test": "perturbed_uniform_angle",
        "variance": "real_projective_plane",
        "T_bath": T_b,
        "nu_bath": nu_bath,
        "prefix": f"{output_root}/T_{T_b:.2f}",
    }
    sim = CFMZNeedleDSMC(
        opts=opts,
        info=info,
        vlasov_force=vlasov_force,
        interaction_energy=interaction_energy_fn,
        comm=MPI.COMM_WORLD,
    )
    sim.run(nsteps=nsteps, monitor_every=nsteps)
    final_var = float(sim.history["circular_var"][-1])
    circular_vars.append(final_var)
    Print(f"    circular_var = {final_var:.4f}")

# ------------------------------------------------------------------ #
# Phase diagram plot                                                   #
# ------------------------------------------------------------------ #
if comm.Get_rank() == 0:
    T_c = 8.0 / np.pi   # spinodal estimate

    fig, ax, _ = fig_axes()
    ax.plot(T_bath_values, circular_vars, "o-", color="black",
            linewidth=1.5, markersize=5, label=r"$\sigma^2$ (DSMC)")
    ax.axvline(T_c, color="gray", linestyle="--", linewidth=1.0,
               label=r"$T_c = 8/\pi \approx 2.55$ (spinodal)")
    ax.set_xlabel(r"$T_{\mathrm{bath}}$")
    ax.set_ylabel(r"circular variance $\sigma^2 = 1 - |\langle e^{2i\theta}\rangle|$")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False)

    # annotate phases
    ax.text(0.5, 0.08, "nematic", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="black")
    ax.text(0.75, 0.92, "isotropic", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="black")

    fig.savefig(f"{output_root}/phase_diagram.pdf")
    fig.savefig(f"{output_root}/phase_diagram.png", dpi=400)
    plt.close(fig)
    Print(f"\nPhase diagram saved to {output_root}/phase_diagram.{{pdf,png}}")

Print("Sweep complete.")
