from .cfmz import CFMZNeedleDSMC
from .boltzmann import BoltzmannDSMC
from petsc4py import PETSc
from mpi4py import MPI

def Print(*args, **kwargs):
    """Helper to print only from rank 0."""
    if PETSc.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)
 
