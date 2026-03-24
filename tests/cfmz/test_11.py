"""
Test with with Vlasov force
----------------------------
Here we consdier a quadratic potential reuslting in the Valsov force
V = -alpha(theta-theta_av)
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMC, Print
import numpy as np
import matplotlib.pyplot as plt

Opt = PETSc.Options()
Print("Running homogeneous CFMZ needle DSMC with options:")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 128)
dt = Opt.getReal("dt", 0.01)
nu = Opt.getReal("nu", 1)
nsteps = Opt.getInt("nsteps", 2000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
monitor_every = Opt.getInt("monitor_every", 10)

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
vlasov_interpolant_mesh = np.linspace(0, 2*np.pi, bins)
def hat_interpolant(x,y, mesh):
    hat = lambda t: 1-np.abs(t)
    interpolant = y[...,None]*hat(x[...,None][:,0]-mesh)
    interpolant = (1/interpolant.shape[0])*interpolant
    return np.sum(interpolant, axis=0)

comm = MPI.COMM_WORLD
def vlasov_force(theta):
    hist, theta_edges = np.histogram(theta, bins=bins)
    hist = comm.allreduce(hist, op=MPI.SUM)
    delta_theta = 2*np.pi/(bins+1)
    normalisation = np.sum(hist)*delta_theta
    hist = hist/normalisation
    L = info["length"]
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    np.sin(theta[...,None]-centers)
    B = np.sign(np.sin(theta[...,None][:,0]-centers))
    A = np.cos(theta[...,None][:,0]-centers)
    force = -np.sum(A*B*hist, axis=1)*delta_theta
    if comm.Get_rank() == 0:
        vlasov_interpolant = hat_interpolant(theta, force, vlasov_interpolant_mesh)
        vlasov_energy_history.append(np.sqrt(np.sum(np.abs(vlasov_interpolant)**2) \
                                    /vlasov_interpolant_mesh[1]-vlasov_interpolant_mesh[0])
                                    )
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        time = np.array(range(len(vlasov_energy_history)))*dt
        ax.plot(time,
            vlasov_energy_history,
            color="black",
            linewidth=1.5,
        )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\mathcal{V}(\theta)|$")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.tight_layout(pad=0.2)
        fig.savefig(f"output/test_11_output_cfmz_{collision_type}/vlasov_energy.pdf", bbox_inches="tight")
        fig.savefig(f"output/test_11_output_cfmz_{collision_type}/vlasov_energy.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
    return (L**2)*force.reshape(force.shape[0],1)

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
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")

