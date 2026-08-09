"""
Thermalization under a non-symmetric transition probability
------------------------------------------------------------
Same setup as test_needle_0 (no Vlasov force, no transport, Maxwell
selection), but the rigid-rod impulse is replaced by the handed (chiral)
exchange rule: the pair vector (sqrt(m/4) g.n, sqrt(I/2) omega_i) is
rotated counterclockwise by chi ~ U(0, pi).  The forward and reverse
transition probabilities differ (detailed balance fails pointwise), while
the reciprocity principle of Cercignani & Lampis holds exactly, so the
H-theorem of arXiv:2508.10744 still predicts relaxation to the Maxwellian
with equipartition.  The observables of interest are the translational
and rotational temperatures, which must converge to a common value.
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
from dsmc import CFMZNeedleDSMCHomo as CFMZNeedleDSMC, Print

Opt = PETSc.Options()
Print("Running homogeneous CFMZ needle DSMC with options:")

nlocal = Opt.getReal("nlocal", 1e6)
nlocal = int(nlocal)
bins = Opt.getInt("bins", 256)
dt = Opt.getReal("dt", 0.05)
nu = Opt.getReal("nu", 10)
nsteps = Opt.getInt("nsteps", 1000)
seed = Opt.getInt("seed", 47)
grazing_collision = Opt.getBool("grazing_collision", False)
collision_type = Opt.getString("collision_type", "nanbu")
extra_collision = Opt.getInt("extra_collision", 0)+1
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
Print(f"  cross_section=maxwell")
Print(f"  collision_rule=chiral")
Print(f"  grazing_collision={grazing_collision}")

Print("--------------------------------------------------------------------")

info = {"inertia": 1.0,
        "mass": 1.0,
        "length": 1.0,
        "ev": 1.0,       # translational restitution
        "om": 1.0,       # rotational restitution
        "cutoff": 0.1,   # angular cutoff (unused by the chiral rule)
        "cross_section": "maxwell",
        "collision_rule": "chiral",
       }
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
    "transport": False,
    "prefix": "output/test_needle_29",
}
sim = CFMZNeedleDSMC(
    opts=opts,
    info=info,
    comm=MPI.COMM_WORLD,
)
sim.run(nsteps=nsteps, monitor_every=monitor_every)
Print("Simulation complete.")
