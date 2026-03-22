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
from dsmc import CFMDSMC, Print
import numpy as np

Opt = PETSc.Options()
Print("Running homogeneous CFM DSMC with options:")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 128)
dt = Opt.getReal("dt", 0.1)
nu = Opt.getReal("nu", 10)
nsteps = Opt.getInt("nsteps", 2000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 10)+1
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
    print("Force mean: ", np.sum(np.abs(force))/len(force))
    print("Force max: ", np.max(np.abs(force)))
    print("Force min: ", np.min(np.abs(force)))
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
    "prefix": "output/test_10",
}
sim = CFMDSMC(
    opts=opts,
    info=info,
    vlasov_force=vlasov_force,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")

