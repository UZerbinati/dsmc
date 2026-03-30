"""
Andersen thermostat — isothermal run, isotropic phase
------------------------------------------------------
Same mean-field setup as test_12 (Onsager potential, L = √12, ν = 4)
but with an Andersen thermostat fixing T_bath = 8.0.

The dimensionless Onsager coupling at this temperature is
    α = L² / T_bath = 12 / 8 = 1.5 ,
well below the spinodal threshold α_c = 3π/2 ≈ 4.71, so thermal
fluctuations dominate and the system remains in the isotropic phase
(circular_var ≈ 1) throughout the run.

Compare with test_26 (nematic phase) and test_27 (full phase diagram).
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
Print("Running isothermal CFMZ needle DSMC — isotropic phase (Andersen thermostat):")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 4)
nsteps = Opt.getInt("nsteps", 1000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0) + 1
monitor_every = Opt.getInt("monitor_every", 100)
T_bath = Opt.getReal("T_bath", 8.0)
nu_bath = Opt.getReal("nu_bath", 4.0)

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
Print(f"  cross_section=maxwell")
Print(f"  grazing_collision={grazing_collision}")
Print(f"  T_bath={T_bath}  (Andersen thermostat)")
Print(f"  nu_bath={nu_bath}")
Print(f"  Onsager coupling α = L²/T_bath = {12.0/T_bath:.3f}  (α_c ≈ 4.71)")

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

vlasov_energy_history = []

_grid_centers = (np.arange(bins) + 0.5) * (2*np.pi / bins)
_diff  = _grid_centers[:, None] - _grid_centers[None, :]
_K_mat = np.sign(np.sin(_diff)) * np.cos(_diff)
_W_mat = np.abs(np.sin(_diff))

comm = MPI.COMM_WORLD
_delta_theta = 2*np.pi / bins
_centers = (np.arange(bins) + 0.5) * _delta_theta

output_dir = f"output/test_25_output_cfmz_{collision_type}"

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
    V_grid = -_delta_theta * (_K_mat @ rho)
    if comm.Get_rank() == 0:
        vlasov_energy_history.append(np.sqrt(np.sum(V_grid**2) * _delta_theta))
        fig, ax, _ = fig_axes()
        time = np.array(range(len(vlasov_energy_history))) * dt
        ax.plot(time, vlasov_energy_history, color="black", linewidth=1.5)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\mathcal{V}(\theta)|$")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"{output_dir}/vlasov_energy.pdf")
        fig.savefig(f"{output_dir}/vlasov_energy.png", dpi=400)
        plt.close(fig)
    return force.reshape(-1, 1)

def interaction_energy_fn(theta):
    rho = _cic_density(theta)
    W = np.abs(np.sin(_centers[:, None] - _centers[None, :]))
    return float(np.sum(W * rho[:, None] * rho[None, :]) * _delta_theta**2)

opts = {
    "nlocal": nlocal,
    "nu": nu,
    "dt": dt,
    "bins": bins,
    "extra_collision": extra_collision,
    "grazing_collision": grazing_collision,
    "collision_type": collision_type,
    "seed": seed,
    "test": "perturbed_uniform_angle",
    "variance": "real_projective_plane",
    "T_bath": T_bath,
    "nu_bath": nu_bath,
    "prefix": "output/test_25",
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
